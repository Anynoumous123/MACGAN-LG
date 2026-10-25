from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch import optim
import config as cf
from dataloader import GeneDataset, create_dataloader
from models.MACGAN_LG import Generator, CDiscriminator, Auxiliary_Classifier, Auxiliary_Classifier2
from train.utils import Train, save_model, current_time


if __name__ == '__main__':
    Path(cf.model.SAVE_PATH).mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    path = Path('D:\\Dataset\\training_set.xlsx')
    dataset = GeneDataset(data_path=path, num_classes=2)
    dataloader = create_dataloader(dataset, cf.dataset.DataLoader)

    gen = Generator(
        in_features=cf.train.LATENT_SIZE,
        out_features=dataset.data_size,
        hidden_layers=cf.model.G_HIDDEN_LAYERS,
    ).to(device)

    dis = CDiscriminator(
        input_features=dataset.data_size,
        hidden_layers=cf.model.D_HIDDEN_LAYERS,
    ).to(device)

    cls = Auxiliary_Classifier(
        num_classes=2,
        input_features=dataset.data_size,
        hidden_layers=cf.model.C_HIDDEN_LAYERS,
    ).to(device)

    cls2 = Auxiliary_Classifier2(
        num_classes=2,
        input_features=dataset.data_size,
        hidden_layers=cf.model.C_HIDDEN_LAYERS,
    ).to(device)

    d_optim = optim.RMSprop(
        dis.parameters(),
        lr=cf.train.DIS_LEARNING_RATE
    )
    g_optim = optim.RMSprop(
        gen.parameters(),
        lr=cf.train.GEN_LEARNING_RATE,
    )
    c_optim = optim.Adam(
        cls.parameters(),
        lr=cf.train.CLS_LEARNING_RATE,
    )
    c2_optim = optim.Adam(
        cls2.parameters(),
        lr=cf.train.CLS_LEARNING_RATE,
    )
    cls_loss_function = nn.BCELoss()
    trainer = Train(
        generator=gen,
        discriminator=dis,
        classifier=cls,
        classifier2=cls2,
        g_optim=g_optim,
        d_optim=d_optim,
        c_optim=c_optim,
        c2_optim=c2_optim,
        device=device,
        class_loss=cls_loss_function
    )

    def label_transform(labels: torch.Tensor) -> torch.Tensor:
        
        transformed_labels = torch.zeros((labels.size(0), 1), dtype=torch.float32)
        
        for i in range(labels.size(0)):
            label = labels[i]
            if torch.equal(label, torch.tensor([1.0])):
                transformed_labels[i] = torch.tensor([0.0], dtype=torch.float32)
            else:
                transformed_labels[i] = torch.tensor([1.0], dtype=torch.float32)

        return transformed_labels

    loss_labels = ['dis_loss', 'gen_loss']
    dloss_gloss_racc_facc = np.zeros((2, cf.train.EPOCHS))

    for epoch in range(cf.train.EPOCHS):
        dis_loss, gen_loss = 0, 0
        for i, (data, labels) in enumerate(dataloader):
            fake_labels = label_transform(labels)
            data, labels, fake_labels = data.to(device), labels.to(device), fake_labels.to(device)
            noise = torch.randn(len(data), cf.train.LATENT_SIZE, device=device, dtype=torch.float32)

            for j in range(0, 3):
                ''' train discriminator with real data '''
                trainer.discriminator_zero_grad()
                if epoch >= 4000:
                    trainer.classifier_zero_grad()
                    trainer.classifier2_zero_grad()
                # train real data
                dis_real_loss = trainer.train_dis_with_real_data_ic(data, labels, fake_labels, epoch)
                ''' train discriminator with fake data '''
                fake_data = gen(noise, labels)
                dis_fake_loss, source_fake_loss, class_loss_df = trainer.train_dis_with_fake_data_ic(fake_data.detach(), labels, fake_labels ,epoch)

                # Train with gradient penalty
                gradient_penalty = trainer.calculate_gradient_penalty(data=data.data, real_labels=labels, fake_data=fake_data.data, out_features=20530,
                                                                  device=device)

                dis_loss = dis_fake_loss - dis_real_loss + gradient_penalty
                dis_loss.backward()
                Wasserstein_D = dis_real_loss - source_fake_loss
                d_optim.step()
                if epoch >= 4000:
                    c_optim.step()
                    c2_optim.step()

            ''' train generator '''
            trainer.generator_zero_grad()
            noise = torch.randn(len(data), cf.train.LATENT_SIZE, device=device, dtype=torch.float32)
            fake_data = gen(noise, labels)
            generator_loss, class_loss_gf = trainer.train_generator_ic(fake_data, labels, fake_labels, epoch)
            gen_loss = -generator_loss
            g_optim.step()

        dis_loss /= len(dataloader)
        gen_loss /= len(dataloader)

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}]')

        print(f"\rGen Loss: {gen_loss:.4f}, Dis Loss: {dis_loss:.4f}, loss_dr: {class_loss_df:.4f}, loss_gf: {class_loss_gf:.4f}, "
                f"WD: {Wasserstein_D:.4f}")

    save_model(gen, Path(cf.model.SAVE_PATH, current_time() + cf.model.GEN_NAME))



