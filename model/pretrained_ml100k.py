import torch
import torch.nn as nn 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle

class genreEncoder(nn.Module):
    def __init__(self):
        super(genreEncoder, self).__init__()
        self.linear1 = nn.Linear(3, 18)
        self.linear2 = nn.Linear(18, 6)
        self.relu = nn.ReLU()
    
    def forward(self, input):
        out = self.linear1(input)
        out = self.relu(out)
        out = self.linear2(out)

        return out

class genreDecoder(nn.Module):
    def __init__(self):
        super(genreDecoder, self).__init__()
        self.linear = nn.Linear(6, 18)
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, input):
        out = self.linear(input)
        out = self.softmax(out)

        return out

class pretrained(nn.Module):
    def __init__(self, encoder, decoder):
        super(pretrained, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

class GenreDataset(Dataset):
    def __init__(self, source, target):
        self.feature = source
        self.label = target
        self.len = len(source)
    
    def __getitem__(self, idx):
        x = self.feature[idx]
        initial_tensor = torch.tensor(x)
        target_tensor = torch.tensor(self.label[idx])

        return initial_tensor, target_tensor

    def __len__(self):
        return self.len

if __name__ == '__main__':
    with open('user_preference', 'rb') as fp:
        user_preference = pickle.load(fp)

    with open('user_init', 'rb') as fp:
        init_user_feature = pickle.load(fp)

    trainset = GenreDataset(init_user_feature, user_preference)

    train_data_loader = DataLoader(trainset, batch_size=32)

    #training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = genreEncoder()
    encoder.to(device)
    decoder = genreDecoder()
    decoder.to(device)

    model = pretrained(encoder, decoder)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3, 
                                 weight_decay=1e-5)
    model.train()
    for epoch in range(10):
        for data in train_data_loader:
            x = data[0].to(device, dtype=torch.float)
            # x = torch.unsqueeze(x, 2)
            y = data[1].to(device, dtype=torch.float)
            # y = torch.unsqueeze(y, 2)
            # print(x.shape)
            reconstruction = model(x)
            
            loss = criterion(reconstruction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')     

    #save
    torch.save(model.state_dict(), '../../../saved_models/ml100k/pretrained_filter.pt')