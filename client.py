from utils import *
from architecture import *

# Implement FedPNAS Client
class Client:
    def __init__(self, architecture, client_id):
        self.client_id = client_id
        self.architecture = architecture
        self.architecture.to(device)
        self.dataset, self.batch_sampler, self.data = None, None, None
        self.loss = nn.NLLLoss()
        self.aux_loss = nn.MSELoss()
        self.base_opt, self.spec_opt, self.aux_opt, self.finetune_opt = self.init_opt()
        self.base_scheduler, self.spec_scheduler, self.aux_scheduler, self.finetune_scheduler = self.init_scheduler()

    def load_data(self, X, Y):
        self.data = TensorDataset(X, Y)
        self.batch_sampler = RandomSampler(self.data)
        self.dataset = DataLoader(self.data, sampler=self.batch_sampler, batch_size=BATCH_SIZE)

    def init_opt(self):
        base_opt = opt.Adam([
            {'params': self.architecture.base_arch.parameters(), 'lr': lr_decay(0, init_learning_rate=2e-3)},
            {'params': self.architecture.base_sampler.parameters(), 'lr': lr_decay(0, init_learning_rate=2e-3)}]
        )

        spec_opt = opt.Adam([
            {'params': self.architecture.spec_arch.parameters(), 'lr': lr_decay(0, init_learning_rate=2e-3)},
            {'params': self.architecture.spec_sampler.parameters(), 'lr': lr_decay(0, init_learning_rate=2e-3)},
        ])

        aux_opt = opt.Adam([
            {'params': self.architecture.base_arch.parameters(), 'lr': lr_decay(0, init_learning_rate=2e-3)},
            {'params': self.architecture.base_sampler.parameters(), 'lr': lr_decay(0, init_learning_rate=2e-3)},
            {'params': self.architecture.aux.parameters(), 'lr': lr_decay(0, init_learning_rate=2e-3)}
        ])

        finetune_opt = opt.Adam([
            {'params': self.architecture.spec_arch.parameters(), 'lr': lr_decay(0, init_learning_rate=1e-3)},
            {'params': self.architecture.spec_sampler.parameters(), 'lr': lr_decay(0, init_learning_rate=1e-5)},
        ])

        return base_opt, spec_opt, aux_opt, finetune_opt

    def init_scheduler(self):
        base_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.base_opt,
            lr_lambda=lambda step: lr_decay(step, init_learning_rate=2e-3) / lr_decay(0, init_learning_rate=2e-3)
        )

        spec_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.spec_opt,
            lr_lambda=lambda step: lr_decay(step, init_learning_rate=2e-3) / lr_decay(0, init_learning_rate=2e-3)
        )

        aux_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.aux_opt,
            lr_lambda=lambda step: lr_decay(step, init_learning_rate=2e-3) / lr_decay(0, init_learning_rate=2e-3)
        )

        finetune_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.finetune_opt,
            lr_lambda=lambda step: lr_decay(step, init_learning_rate=1e-4) / lr_decay(0, init_learning_rate=1e-4)
        )

        return base_scheduler, spec_scheduler, aux_scheduler, finetune_scheduler

    def local_finetune(self, server_model, n_sample, n_batch=10):
        self.architecture.load_state_dict(deepcopy(server_model.state_dict()))
        freeze(self.architecture.aux)
        freeze(self.architecture.base_arch)
        freeze(self.architecture.base_sampler)
        tqdm_loader = trange(n_batch, leave='True', desc=f'Client {self.client_id}')
        for _ in tqdm_loader:
            X, Y = next(iter(self.dataset))
            X, Y = X.to(device), Y.to(device)
            # Compute standard loss
            self.finetune_opt.zero_grad()
            X_spec = self.architecture(X, n_sample, use_spec=True, use_aux=False)
            loss = self.loss(torch.log(X_spec), Y)
            loss.backward()
            self.finetune_opt.step()
        self.finetune_scheduler.step()

    def std_loss(self, X, Y, n_sample, use_aux=False, compute_acc=False):
        X_spec = self.architecture(X, n_sample=n_sample, use_spec=True, use_aux=False)
        std_loss = self.loss(torch.log(X_spec), Y)
        acc = 0.0
        if compute_acc:
            acc = compute_accuracy(X_spec, Y)
        if use_aux:
            X_aux = self.architecture(X, n_sample=n_sample, use_spec=False, use_aux=True)
            aux_loss = self.aux_loss(X_spec, X_aux)
            if compute_acc:
                return std_loss, aux_loss, acc
            else:
                return std_loss, aux_loss
        else:
            if compute_acc:
                return std_loss, acc
            else:
                return std_loss

    def batch_update(self, batch, X, Y, n_sample, server_model, verbose=False, meta=True):
        X, Y = X.to(device), Y.to(device)
        if verbose:
            print(f'Local update batch {batch}')
        if meta:
            return self.batch_meta_update(X, Y, n_sample, server_model, verbose=verbose)
        else:
            return self.batch_std_update(X, Y, n_sample, verbose=verbose)

    def batch_std_update(self, X, Y, n_sample, verbose=False):
        # Compute standard loss
        self.base_opt.zero_grad()
        self.spec_opt.zero_grad()
        loss, aux_loss, acc = self.std_loss(X, Y, n_sample, use_aux=True, compute_acc=True)
        std_loss = loss + 0.5 * aux_loss
        std_loss.backward()
        if verbose:
            print(f'Loss = {loss.item()}, Aux Loss = {aux_loss.item()}, Std Loss = {std_loss.item()}')
        self.base_opt.step()
        self.spec_opt.step()
        return loss, aux_loss, std_loss, acc

    def batch_meta_update(self, X, Y, n_sample, server_model, verbose=False):
        # Compute standard loss
        self.base_opt.zero_grad()
        self.spec_opt.zero_grad()
        loss, aux_loss = self.std_loss(X, Y, n_sample, use_aux=True)
        loss = loss + 0.5 * aux_loss
        loss.backward()
        if verbose:
            print(f'Standard Loss = {loss.item()}')

        # Extract loss gradient for personalized parameters
        dL_d_spec_arch, dL_d_spec_sampler = [], []
        for p in self.architecture.spec_arch.parameters():
            if p.grad is not None:
                dL_d_spec_arch.append(p.grad)
            else:
                dL_d_spec_arch.append(torch.zeros(p.shape).to(device))
        for p in self.architecture.spec_sampler.parameters():
            if p.grad is not None:
                dL_d_spec_sampler.append(p.grad)
            else:
                dL_d_spec_sampler.append(torch.zeros(p.shape).to(device))
        if verbose:
            print(f'Extract Loss Gradient for Personalized Params')

        # Reset gradient and step personalized parameters
        self.spec_opt.step()
        self.base_opt.zero_grad()
        self.spec_opt.zero_grad()
        if verbose:
            print(f'Reset Gradient and Step Personalized Params')

        # Compute meta loss
        loss, aux_loss, acc = self.std_loss(X, Y, n_sample, use_aux=True, compute_acc=True)
        meta_loss = loss + 0.5 * aux_loss
        meta_loss.backward()
        if verbose:
            print(f'Meta Loss = {meta_loss.item()}')

        # Reset personalized parameters to server
        self.architecture.spec_arch.load_state_dict(deepcopy(server_model.spec_arch.state_dict()))
        self.architecture.spec_sampler.load_state_dict(deepcopy(server_model.spec_sampler.state_dict()))
        if verbose:
            print(f'Reset Personalized Params to Server Params')

        # Compute meta loss gradient for personalized parameters
        grad_product = 0.0
        for i, p in enumerate(self.architecture.spec_arch.parameters()):
            if p.grad is not None:
                grad_product += torch.dot(dL_d_spec_arch[i].view(-1), p.grad.view(-1))

        for i, p in enumerate(self.architecture.spec_arch.parameters()):
            if p.grad is not None:
                p.grad -= self.spec_scheduler.get_last_lr()[0] * dL_d_spec_arch[i] * grad_product
        if verbose:
            print(f'Compute Meta Loss Gradient for Spec Arch')

        grad_product = 0.0
        for i, p in enumerate(self.architecture.spec_sampler.parameters()):
            if p.grad is not None:
                grad_product += torch.dot(dL_d_spec_sampler[i].view(-1), p.grad.view(-1))
        for i, p in enumerate(self.architecture.spec_sampler.parameters()):
            if p.grad is not None:
                p.grad -= self.spec_scheduler.get_last_lr()[1] * dL_d_spec_sampler[i] * grad_product
        if verbose:
            print(f'Compute Meta Loss Gradient for Spec Sampler')

        self.base_opt.step()
        self.spec_opt.step()
        return loss, aux_loss, meta_loss, acc

    def local_update(self, server_model, n_sample, n_batch=10, meta=True):
        freeze(self.architecture.aux)
        self.architecture.load_state_dict(deepcopy(server_model.state_dict()))

        if n_batch == -1:
            tqdm_loader = tqdm(self.dataset, leave='True', desc=f'Client {self.client_id}')
            for batch, (X, Y) in enumerate(tqdm_loader):
                loss, aux_loss, meta_loss, acc = self.batch_update(batch, X, Y, n_sample, server_model, meta=meta)
                tqdm_loader.set_postfix_str(
                    f'Loss = {meta_loss.item():.3f}, Acc={acc:.3f}'
                )

        else:
            tqdm_loader = trange(n_batch, leave='True', desc=f'Client {self.client_id}')
            for batch in tqdm_loader:
                X, Y = next(iter(self.dataset))
                loss, aux_loss, meta_loss, acc = self.batch_update(batch, X, Y, n_sample, server_model, meta=meta)
                tqdm_loader.set_postfix_str(
                    f'Loss = {meta_loss.item():.3f}, Acc={acc:.3f}'
                )

        self.base_scheduler.step()
        self.spec_scheduler.step()

    def batch_aux_update(self, X, Y, n_sample, compute_acc=False):
        X, Y = X.to(device), Y.to(device)
        # Compute standard loss
        self.aux_opt.zero_grad()
        X_aux = self.architecture.forward(X, n_sample=n_sample, use_spec=False, use_aux=True)

        loss = self.loss(torch.log(X_aux), Y)
        loss.backward()
        self.aux_opt.step()

        if compute_acc:
            acc = compute_accuracy(X_aux, Y)
            return loss, acc
        else:
            return loss

    def aux_update(self, server_model, n_sample, n_batch=10):
        self.architecture.load_state_dict(deepcopy(server_model.state_dict()))
        if n_batch == -1:
            tqdm_loader = tqdm(self.dataset, leave='True', desc=f'Client {self.client_id}')
            for batch, (X, Y) in enumerate(tqdm_loader):
                loss, acc = self.batch_aux_update(X, Y, n_sample, compute_acc=True)
                tqdm_loader.set_postfix_str(f'Loss={loss.item():.3f}, Acc={acc:.3f}')

        else:
            tqdm_loader = trange(n_batch, leave='True', desc=f'Client {self.client_id}')
            for batch in tqdm_loader:
                X, Y = next(iter(self.dataset))
                loss, acc = self.batch_aux_update(X, Y, n_sample, compute_acc=True)
                tqdm_loader.set_postfix_str(f'Loss={loss.item():.3f}, Acc={acc:.3f}')

        self.base_scheduler.step()
        self.spec_scheduler.step()


# Implement FedAvg Client
class NaiveClient:
    def __init__(self, architecture, client_id):
        self.client_id = client_id
        self.architecture = architecture
        self.architecture.to(device)
        self.dataset, self.batch_sampler, self.data = None, None, None
        self.loss = nn.NLLLoss()
        self.opt = self.init_opt()
        self.scheduler = self.init_scheduler()

    def load_data(self, X, Y):
        self.data = TensorDataset(X, Y)
        self.batch_sampler = RandomSampler(self.data)
        self.dataset = DataLoader(self.data, sampler=self.batch_sampler, batch_size=BATCH_SIZE)

    def init_opt(self):
        return opt.Adam(
            [{'params': self.architecture.parameters(), 'lr': lr_decay(0, init_learning_rate=2e-3)}]
        )

    def init_scheduler(self):
        return torch.optim.lr_scheduler.LambdaLR(
            self.opt,
            lr_lambda=lambda step: lr_decay(step, init_learning_rate=2e-3) / lr_decay(0, init_learning_rate=2e-3)
        )

    def batch_update(self, X, Y):
        self.opt.zero_grad()
        X, Y = X.to(device), Y.to(device)
        X_spec = self.architecture(X)
        loss = self.loss(torch.log(X_spec), Y)
        acc = torch.sum(torch.eq(torch.argmax(X_spec, dim=1), Y))
        loss.backward()
        self.opt.step()
        return loss, acc

    def local_update(self, server_model, n_batch=10):
        self.architecture.load_state_dict(deepcopy(server_model.state_dict()))

        if n_batch == -1:
            tqdm_loader = tqdm(self.dataset, leave='True', desc=f'Client {self.client_id}')
            for _, (X, Y) in enumerate(tqdm_loader):
                loss, acc = self.batch_update(X, Y)
                tqdm_loader.set_postfix_str(f'Loss = {loss.item():.3f}, Acc={acc:.3f}')

        else:
            tqdm_loader = trange(n_batch, leave='True', desc=f'Client {self.client_id}')
            for _ in tqdm_loader:
                X, Y = next(iter(self.dataset))
                loss, acc = self.batch_update(X, Y)
                tqdm_loader.set_postfix_str(f'Loss = {loss.item():.3f}, Acc={acc:.3f}')

        self.scheduler.step()

