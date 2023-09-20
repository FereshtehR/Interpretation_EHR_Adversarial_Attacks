import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from attack_core import Attack


### Generate perturbation on each iteration
class CW(Attack):

    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01, device='cpu', gamma=1, attack_type=1):
        super().__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.device = device
        self._supported_mode = ['default', 'targeted']
        self.gamma = gamma
        self.attack_type = attack_type

    def forward(self, images, labels, lengths, indices, target_cont, d_target_cont):
        """
        Overridden.
        """
        images_in_the_graph = images.clone()
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        target_class = 1 - labels

        org_outputs, org_alpha, org_beta, org_context = self.model(images, lengths, get_context=True)
        org_outputs_detached = org_outputs.detach()
        W_emb = self.model.get_Wemb()
        W = self.model.get_W()
        cont_org = org_alpha * torch.matmul(W, org_beta.transpose(0, 1).transpose(1, 2) * W_emb).transpose(0, 1)[labels] * images
        cont_clean = cont_org.detach()
        self._targeted = True

        if self._targeted:
            target_labels = labels


        Flatten = nn.Flatten()

        ### loss Function Coefficient
        initial_f_coeff = 1
        f_coeff = initial_f_coeff

        ### Optimization coefficient
        initial_opt_coeff = 1
        opt_coeff = initial_opt_coeff

        ### variable regarding adjusting the coefficients using Penalty
        last_iter_change = 0
        penalty_increase = 0

        ### Initializing adversarial example using the original clean sample
        best_adv_images = images.clone().detach()
        modifier = torch.zeros(images.size()).float()
        modifier = torch.normal(mean=modifier, std=0.001).to(self.device)
        modifier.requires_grad = True
        adv_images = modifier + images

        ### Initialize optimizer
        optimizer = optim.SGD([modifier], lr=0.005)


        #### Generate perturbations step by step
        for step in range(self.steps):

            # Calculate loss
            diff = torch.abs(Flatten(modifier))
            current_L1 = diff
            L1_loss = current_L1.sum()

            outputs, alpha, beta, adv_context = self.model(adv_images, lengths, get_context=True)
            W_emb = self.model.get_Wemb()
            W = self.model.get_W()
            cont = alpha * torch.matmul(W, beta.transpose(0, 1).transpose(1, 2) * W_emb).transpose(0, 1)[1] * adv_images
            diff_cont = torch.abs(Flatten(cont) - Flatten(target_cont))
            current_Cont = diff_cont
            cont_loss = current_Cont.sum()


            if self._targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()


            ### Calculate KL loss
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            input_kl = F.log_softmax(outputs, dim=1)
            target_kl = F.softmax(org_outputs_detached, dim=1)
            kld_loss = kl_loss(input_kl, target_kl)
            dist_coeff = self.gamma

            ### Select the attack
            if self.attack_type == 2:  # KL
                cost = opt_coeff * cont_loss + f_coeff * kld_loss + dist_coeff * L1_loss
            elif self.attack_type == 3:  # confident
                self.kappa = 0.8
                cost = opt_coeff * cont_loss + f_coeff * f_loss + dist_coeff * L1_loss
            else:  # original
                self.kappa = 0.0
                cost = opt_coeff * cont_loss + f_coeff * f_loss + dist_coeff * L1_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # ISTA (ignore the noise if very small)
            noise = modifier.clone().detach()
            lamb = 0.0001
            cond1 = torch.greater(noise, lamb).type(torch.float32)
            cond2 = torch.less_equal(torch.abs(noise), lamb).type(torch.float32)
            cond3 = torch.less(noise, -lamb).type(torch.float32)
            temp_modifier = torch.multiply(cond1, noise) + torch.multiply(cond3, noise) + torch.multiply(cond2, torch.add(noise, -noise))
            modifier.data = temp_modifier.data
            adv_images = modifier + images
            adv_images = torch.clamp(adv_images, min=0.0, max=1.0)
            best_adv_images = adv_images.detach()
            outputs_best, alpha_best, beta_best = self.model(best_adv_images, lengths)
            _, pre = torch.max(outputs_best.detach(), 1)
            correct = (pre == labels).item()
            cont_best_adv = alpha_best * torch.matmul(W, beta_best.transpose(0, 1).transpose(1, 2) * W_emb).transpose(0, 1)[labels] * best_adv_images


            ### Apply Dynamic Pentaly
            if f_coeff != initial_f_coeff:
                if correct:
                    if step - last_iter_change > 5:
                        penalty_increase = 0
                        f_coeff = initial_f_coeff
                        opt_coeff = initial_opt_coeff

                if not correct:
                    f_coeff = f_coeff * 2
                    opt_coeff /= 2
                    penalty_increase += 1

                    if penalty_increase >= 10:
                        print(f"Penalty is too large : step {step}")


        return best_adv_images, cont_clean, cont_best_adv


