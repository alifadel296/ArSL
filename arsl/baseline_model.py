
from torch import nn



class BaselineModel (nn.Module):
  def __init__ (self, conv_size, stride , lstm_input , hidden_size , num_layers):
    super(BaselineModel , self).__init__()
    #Conv Layers
    self.backBone = nn.Sequential()
    
    for i in range(len(conv_size) - 1):
      
      self.backBone.append(
        nn.Conv2d(
          conv_size[i],
          conv_size[i + 1],
          3,
          padding = 0 if stride[i]== 2 else 1,
          stride= stride[i])
        )
      
      self.backBone.append(nn.BatchNorm2d(conv_size[i + 1]))
      
      self.backBone.append(nn.ReLU())
      
    #AdaptiveAvgPooling Layer to reduce the size of output to 1*1*512
    
    self.pooling = nn.AdaptiveAvgPool2d(1)
    
 
    self.lstm = nn.LSTM(
      input_size=lstm_input,
      hidden_size=hidden_size,
      num_layers=num_layers, 
      batch_first=True
      )  # the input Shape is (batch, seq_len, feature_number)
    
    self.classifier = nn.Linear(hidden_size, 502)
    
  def forward(self , x):
    
    batch_size , num_frames , h , w , c = x.shape
    x = x.view(batch_size*num_frames , h ,w , c)
    
    # Run the BackBone block then reduce the size to 1*1*512
    x = self.backBone(x)
    x = self.pooling(x)
    
    # Reshape
    x = x.view(batch_size , num_frames , -1)
    
    lstm_out, _ = self.lstm(x)
    x = lstm_out[:, -1, :]
    x = self.classifier(x)
    
    return x
  