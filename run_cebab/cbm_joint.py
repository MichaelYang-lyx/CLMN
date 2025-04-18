import torch
import transformers
from gensim.models import FastText
from torch.optim.lr_scheduler import StepLR
from transformers import RobertaTokenizer, RobertaModel,BertModel, BertTokenizer,GPT2Model, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os
from cbm_template_models import MLP, FC
from cbm_models import ModelXtoC_function, ModelCtoY_function,ModelXtoCtoY_function,Model_ConceptReasoning
from tools import categorize_tensor
# Enable concept or not
mode = 'joint'

# Define the paths to the dataset and pretrained model
# model_name = "microsoft/deberta-base"
model_name = 'lstm' # 'bert-base-uncased' / 'roberta-base' / 'gpt2' / 'lstm'
data_type = "aug_cebab_yelp" # "pure_cebab"/"aug_cebab"/"aug_yelp"/"aug_cebab_yelp"
# Define the maximum sequence length and batch size

max_len = 512
batch_size = 8
lambda_XtoC = 0.5 # lambda > 0 现在没用了

is_aux_logits = False
num_labels = 5 #label的个数
num_epochs = num_epochs_train =25
learning_rate=1e-2
num_each_concept_classes = 3 #每个concept有几个类
concept_loss_weight=100
y2_weight=0
# Load the tokenizer and pretrained model
if model_name == 'roberta-base':
    # tokenizer = RobertaTokenizer.from_pretrained(model_name)
    # model = RobertaModel.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained('/mnt/llm_test_afs/code/yyb/kejieshi/models/roberta-base')
    model = RobertaModel.from_pretrained('/mnt/llm_test_afs/code/yyb/kejieshi/models/roberta-base')
elif model_name == 'bert-base-uncased' :
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained('/mnt/llm_test_afs/code/yyb/kejieshi/models/bert-base-uncased')
    model = BertModel.from_pretrained('/mnt/llm_test_afs/code/yyb/kejieshi/models/bert-base-uncased')
elif model_name == 'gpt2':
    
    model = GPT2Model.from_pretrained('/mnt/llm_test_afs/code/yyb/kejieshi/models/gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('/mnt/llm_test_afs/code/yyb/kejieshi/models/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # Initialize the classification model
    # model = GPT2Classifier(model)   
elif model_name == 'lstm':
    fasttext_model = FastText.load_fasttext_format('/mnt/llm_test_afs/code/yyb/kejieshi/models/fasttext/cc.en.300.bin')
    tokenizer = BertTokenizer.from_pretrained('/mnt/llm_test_afs/code/yyb/kejieshi/models/bert-base-uncased')
    class BiLSTMWithDotAttention(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            embeddings = fasttext_model.wv.vectors
            self.embedding.weight = torch.nn.Parameter(torch.tensor(embeddings))
            self.embedding.weight.requires_grad = False
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers = 1, bidirectional=True, batch_first=True)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim*2, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2)
        )

        def forward(self, input_ids, attention_mask):
            input_lengths = attention_mask.sum(dim=1)
            embedded = self.embedding(input_ids)
            output, _ = self.lstm(embedded)
            weights = F.softmax(torch.bmm(output, output.transpose(1, 2)), dim=2)
            attention = torch.bmm(weights, output)
            logits = self.classifier(attention.mean(1))
            return logits

    model = BiLSTMWithDotAttention(len(tokenizer.vocab), 300, 128)


# Load data
if data_type == "pure_cebab":
    num_concept_labels = 4
    train_split = "train_exclusive"
    test_split = "test"
    val_split = "validation"
    CEBaB = load_dataset("CEBaB/CEBaB")
elif data_type == "aug_cebab":
    num_concept_labels = 10
    train_split = "train_aug_cebab"
    test_split = "test_aug_cebab"
    val_split = "val_aug_cebab"
    CEBaB = {}
    CEBaB[train_split] = pd.read_csv("../dataset/cebab/train_cebab_new_concept_single.csv")
    CEBaB[test_split] = pd.read_csv("../dataset/cebab/test_cebab_new_concept_single.csv")
    CEBaB[val_split] = pd.read_csv("../dataset/cebab/dev_cebab_new_concept_single.csv")
elif data_type == "aug_yelp":
    num_concept_labels = 10
    train_split = "train_aug_yelp"
    test_split = "test_aug_yelp"
    val_split = "val_aug_yelp"
    CEBaB = {}
    CEBaB[train_split] = pd.read_csv("../dataset/cebab/train_yelp_exclusive_new_concept_single.csv")
    CEBaB[test_split] = pd.read_csv("../dataset/cebab/test_yelp_new_concept_single.csv")
    CEBaB[val_split] = pd.read_csv("../dataset/cebab/dev_yelp_new_concept_single.csv")
elif data_type == "aug_cebab_yelp":
    num_concept_labels = 10
    train_split = "train_aug_cebab_yelp"
    test_split = "test_aug_cebab_yelp"
    val_split = "val_aug_cebab_yelp"
    train_split_cebab = pd.read_csv("../dataset/cebab/train_cebab_new_concept_single.csv")
    test_split_cebab = pd.read_csv("../dataset/cebab/test_cebab_new_concept_single.csv")
    val_split_cebab = pd.read_csv("../dataset/cebab/dev_cebab_new_concept_single.csv")
    train_split_yelp = pd.read_csv("../dataset/cebab/train_yelp_exclusive_new_concept_single.csv")
    test_split_yelp = pd.read_csv("../dataset/cebab/test_yelp_new_concept_single.csv")
    val_split_yelp = pd.read_csv("../dataset/cebab/dev_yelp_new_concept_single.csv")
    CEBaB = {}
    CEBaB[train_split] = pd.concat([train_split_cebab, train_split_yelp], ignore_index=True)
    CEBaB[test_split] = pd.concat([test_split_cebab, test_split_yelp], ignore_index=True)
    CEBaB[val_split] = pd.concat([val_split_cebab, val_split_yelp], ignore_index=True)
# Define a custom dataset class for loading the data

class MyDataset(Dataset):
    # Split = train/dev/test
    def __init__(self, split, skip_class = "no majority"):
        self.data = CEBaB[split]
        self.labels = self.data["review_majority"]
        self.text = self.data["description"]
       
        self.food_aspect = self.data["food_aspect_majority"]
        self.ambiance_aspect = self.data["ambiance_aspect_majority"]
        self.service_aspect = self.data["service_aspect_majority"]
        self.noise_aspect =self.data["noise_aspect_majority"]

        if data_type != "pure_cebab":
            # cleanliness price	location	menu variety	waiting time	waiting area	## parking	wi-fi	kids-friendly
            self.cleanliness_aspect = self.data["cleanliness"]
            self.price_aspect = self.data["price"]
            self.location_aspect = self.data["location"]
            self.menu_variety_aspect = self.data["menu variety"]
            self.waiting_time_aspect =self.data["waiting time"]
            self.waiting_area_aspect =self.data["waiting area"]

        self.map_dict = {"Negative":0, "Positive":2, "unknown":1, "":1,"no majority":1}

        self.skip_class = skip_class
        if skip_class is not None:
            self.indices = [i for i, label in enumerate(self.labels) if label != skip_class]
        else:
            self.indices = range(len(self.labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        text = self.text[self.indices[index]]
        label = int(self.labels[self.indices[index]]) - 1

        # gold labels
        food_concept = self.map_dict[self.food_aspect[self.indices[index]]]
        ambiance_concept = self.map_dict[self.ambiance_aspect[self.indices[index]]]
        service_concept = self.map_dict[self.service_aspect[self.indices[index]]]
        noise_concept = self.map_dict[self.noise_aspect[self.indices[index]]]
        
        if data_type != "pure_cebab":
            # noisy labels
            #cleanliness price	location	menu variety	waiting time	waiting area	## parking	wi-fi	kids-friendly
            cleanliness_concept = self.map_dict[self.cleanliness_aspect[self.indices[index]]]
            price_concept = self.map_dict[self.price_aspect[self.indices[index]]]
            location_concept = self.map_dict[self.location_aspect[self.indices[index]]]
            menu_variety_concept = self.map_dict[self.menu_variety_aspect[self.indices[index]]]
            waiting_time_concept = self.map_dict[self.waiting_time_aspect[self.indices[index]]]
            waiting_area_concept = self.map_dict[self.waiting_area_aspect[self.indices[index]]]

        if data_type != "pure_cebab":
            concept_labels = [food_concept,ambiance_concept,service_concept,noise_concept,cleanliness_concept,price_concept,location_concept,menu_variety_concept,waiting_time_concept,waiting_area_concept]
        else: 
            concept_labels = [food_concept,ambiance_concept,service_concept,noise_concept]

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        if data_type != "pure_cebab":
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
                "food_concept": torch.tensor(food_concept, dtype=torch.long),
                "ambiance_concept": torch.tensor(ambiance_concept, dtype=torch.long),
                "service_concept": torch.tensor(service_concept, dtype=torch.long),
                "noise_concept": torch.tensor(noise_concept, dtype=torch.long),
                "cleanliness_concept": torch.tensor(cleanliness_concept, dtype=torch.long),
                "price_concept": torch.tensor(price_concept, dtype=torch.long),
                "location_concept": torch.tensor(location_concept, dtype=torch.long),
                "menu_variety_concept": torch.tensor(menu_variety_concept, dtype=torch.long),
                "waiting_time_concept": torch.tensor(waiting_time_concept, dtype=torch.long),
                "waiting_area_concept": torch.tensor(waiting_area_concept, dtype=torch.long),
                "concept_labels": torch.tensor(concept_labels, dtype=torch.long)
            }
        else:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
                "food_concept": torch.tensor(food_concept, dtype=torch.long),
                "ambiance_concept": torch.tensor(ambiance_concept, dtype=torch.long),
                "service_concept": torch.tensor(service_concept, dtype=torch.long),
                "noise_concept": torch.tensor(noise_concept, dtype=torch.long),
                "concept_labels": torch.tensor(concept_labels, dtype=torch.long)
            }


# Load the data
train_dataset = MyDataset(train_split)
test_dataset = MyDataset(test_split)
val_dataset = MyDataset(val_split)

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

#Set ModelXtoCtoY_layer
    # concept_classes 每个concept有几类；    label_classes  label的个数；  n_attributes concept的个数； n_class_attr 每个concept有几类；
if model_name == 'lstm':
    ModelXtoCtoY_layer =Model_ConceptReasoning(concept_classes = num_each_concept_classes, label_classes = num_labels, n_attributes = num_concept_labels, bottleneck = True, expand_dim = 0, n_class_attr=num_each_concept_classes, use_relu=False, use_sigmoid=False,Lstm=True,aux_logits=is_aux_logits)
    #ModelXtoCtoY_layer = ModelXtoCtoY_function(concept_classes = num_each_concept_classes, label_classes = num_labels, n_attributes = num_concept_labels, bottleneck = True, expand_dim = 0, n_class_attr=num_each_concept_classes, use_relu=False, use_sigmoid=False,Lstm=True,aux_logits=is_aux_logits)
else:
    ModelXtoCtoY_layer= Model_ConceptReasoning(concept_classes = num_each_concept_classes, label_classes = num_labels, n_attributes = num_concept_labels, bottleneck = True, expand_dim = 0, n_class_attr=num_each_concept_classes, use_relu=False, use_sigmoid=False,aux_logits=is_aux_logits)
    #ModelXtoCtoY_layer = ModelXtoCtoY_function(concept_classes = num_each_concept_classes, label_classes = num_labels, n_attributes = num_concept_labels, bottleneck = True, expand_dim = 0, n_class_attr=num_each_concept_classes, use_relu=False, use_sigmoid=False,aux_logits=is_aux_logits)

# Set up the optimizer and loss function
# optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)
optimizer = torch.optim.Adam(list(model.parameters()) + list(ModelXtoCtoY_layer.parameters()), lr=learning_rate)
if model_name == 'lstm':
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# classifier.to(device)
ModelXtoCtoY_layer.to(device)

## 多卡---------------
# if torch.cuda.device_count() > 1:
#     print(f"使用 {torch.cuda.device_count()} 个 GPU!")
#     model = torch.nn.DataParallel(model)
## ------------------

model.to(device)
count=0
loss_list=[]
for epoch in range(num_epochs):
    predicted_concepts_train = []
    predicted_concepts_train_label = []
    ModelXtoCtoY_layer.train()
    model.train()
    single_loss_list=[]
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        food_concept = batch["food_concept"].to(device)
        ambiance_concept=batch["ambiance_concept"].to(device)
        service_concept=batch["service_concept"].to(device)
        noise_concept=batch["noise_concept"].to(device)

        if data_type != "pure_cebab":
            cleanliness_concept = batch["cleanliness_concept"].to(device)
            price_concept = batch["price_concept"].to(device)
            location_concept = batch["location_concept"].to(device)
            menu_variety_concept = batch["menu_variety_concept"].to(device)
            waiting_time_concept = batch["waiting_time_concept"].to(device)
            waiting_area_concept = batch["waiting_area_concept"].to(device)                
        concept_labels=batch["concept_labels"].to(device)
        concept_labels = torch.t(concept_labels)
        concept_labels = concept_labels.contiguous().view(-1) 


        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        if model_name == 'lstm':
            pooled_output = outputs
        else:
            pooled_output = outputs.last_hidden_state.mean(1)  
        outputs, y_pred_2, c_emb_2, c_pred_2 = ModelXtoCtoY_layer(pooled_output)  
        XtoC_output = outputs [1:] 
        XtoY_output = outputs [0:1]
        # XtoC_loss
        # XtoC_logits = torch.nn.Sigmoid()(torch.cat(XtoC_output, dim=0)) # 32*4 00000000111111112222222233333333
        # XtoC_loss = loss_fn(XtoC_logits, concept_labels)

        # XtoY_loss
        # 11111111111111111111111111111
        XtoY_loss = loss_fn(XtoY_output[0], label)
        XtoY_loss_2 = loss_fn(y_pred_2, label)
        #loss = XtoC_loss*lambda_XtoC+XtoY_loss+XtoY_loss_2*0.1

        #concept_loss_2
        mapped_labels = concept_labels.float() * 0.5
        c_pred_2_transposed_flat = c_pred_2.transpose(0, 1).contiguous().view(-1, 1)
        concept_loss  = F.mse_loss(c_pred_2_transposed_flat.squeeze(1), mapped_labels)
        loss = XtoY_loss+XtoY_loss_2*y2_weight+concept_loss*concept_loss_weight
       
        #loss = XtoC_loss
        single_loss_list.append(concept_loss.item())
        loss.backward()
        optimizer.step()
    

        # count+=1
        # if count==100:
            
        #     # print(XtoC_logits)
        #     # concept_predictions = torch.argmax(XtoC_logits, axis=1)
        #     # print(concept_predictions)
        #     # print(concept_labels)

        #     print(y_pred_2)
        #     print(label)
        #     count=0
    loss_list.append(np.mean(single_loss_list))
    print(loss_list)
    model.eval()
    ModelXtoCtoY_layer.eval()
    val_accuracy = 0.
    val_accuracy2 = 0.
    # concept_val_accuracy = 0.
    concept_val_accuracy_2 = 0.
    same_ratio=0.
    test_accuracy = 0.
    test_accuracy2 = 0.

    concept_test_accuracy = 0.
    concept_test_accuracy_2 = 0.
    best_acc_score = 0
    predict_labels = np.array([])
    predict_labels_2 = np.array([])
    true_labels = np.array([])
    concept_predict_labels = np.array([])
    concept_predict_labels_2 = np.array([])
    concept_true_labels = np.array([])
    predict_concepts = []
    

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            food_concept = batch["food_concept"].to(device)
            ambiance_concept=batch["ambiance_concept"].to(device)
            service_concept=batch["service_concept"].to(device)
            noise_concept=batch["noise_concept"].to(device)
            if data_type != "pure_cebab":
                cleanliness_concept = batch["cleanliness_concept"].to(device)
                price_concept = batch["price_concept"].to(device)
                location_concept = batch["location_concept"].to(device)
                menu_variety_concept = batch["menu_variety_concept"].to(device)
                waiting_time_concept = batch["waiting_time_concept"].to(device)
                waiting_area_concept = batch["waiting_area_concept"].to(device)        
            concept_labels=batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels)
            concept_labels = concept_labels.contiguous().view(-1)

            #222222222222222
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if model_name == 'lstm':
                pooled_output = outputs
            else:
                pooled_output = outputs.last_hidden_state.mean(1)  
            outputs,y_pred_2 ,c_emb_2, c_pred_2= ModelXtoCtoY_layer(pooled_output)  
            
            XtoC_output = outputs [1:] 
            XtoY_output = outputs [0:1]         
            predictions = torch.argmax(XtoY_output[0], axis=1)
            val_accuracy += torch.sum(predictions == label).item()

            y_pred_2_res = torch.argmax(y_pred_2, axis=1)
            val_accuracy2 += torch.sum(y_pred_2_res == label).item()
            predict_labels_2 = np.append(predict_labels_2, y_pred_2_res.cpu().numpy())

            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())
            #concept accuracy
            # XtoC_logits = torch.cat(XtoC_output, dim=0)
            # concept_predictions = torch.argmax(XtoC_logits, axis=1)
            
            # concept_val_accuracy += torch.sum(concept_predictions == concept_labels).item()
            # 用 concept2 =========
            cp2=categorize_tensor(c_pred_2)
            concept_predictions_2 = categorize_tensor(c_pred_2).transpose(0, 1).contiguous().view(-1, 1).squeeze(1)
            concept_val_accuracy_2 += torch.sum(concept_predictions_2 == concept_labels).item()
                # 看两个是否 sanme
            # same_ratio+=torch.sum(concept_predictions_2 ==concept_predictions).item()
                # =======
            concept_predict_labels_2 = np.append(concept_predict_labels_2, concept_predictions_2.cpu().numpy())
            concept_predictions_2 = concept_predictions_2.reshape(-1,num_concept_labels) 
            # ==========
            # concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())
            # concept_predictions = concept_predictions.reshape(-1,num_concept_labels)  # reshape 二维向量[batch_size*num_concept_labels]
            
        print(cp2)
        # print(concept_predictions)
        
        print(batch["concept_labels"].to(device))
        val_accuracy /= len(val_dataset)
        val_accuracy2 /= len(val_dataset)
        num_labels = len(np.unique(true_labels))

        # concept_val_accuracy /= len(val_dataset)
        concept_val_accuracy_2 /= len(val_dataset)
        same_ratio/= len(val_dataset)
        concept_num_true_labels = len(np.unique(concept_true_labels))
        
        
        print(c_pred_2)
        #import pdb;pdb.set_trace()
        # print(concept_true_labels)

        macro_f1_scores = []
        for label in range(num_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)

        macro_f1_scores_2 = []
        for label in range(num_labels):
            label_pred = np.array(predict_labels_2) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores_2.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score_2 = np.mean(macro_f1_scores_2)

        # concept_macro_f1_scores = []
        # for concept_label in range(concept_num_true_labels):
        #     concept_label_pred = np.array(concept_predict_labels) == concept_label
        #     concept_label_true = np.array(concept_true_labels) == concept_label
        #     concept_macro_f1_scores.append(f1_score(concept_label_true, concept_label_pred, average='macro'))
        #     concept_mean_macro_f1_score = np.mean(concept_macro_f1_scores)

        concept_macro_f1_scores_2 = []
        for concept_label in range(concept_num_true_labels):
            concept_label_pred_2 = np.array(concept_predict_labels_2) == concept_label
            concept_label_true = np.array(concept_true_labels) == concept_label
            concept_macro_f1_scores_2.append(f1_score(concept_label_true, concept_label_pred_2, average='macro'))
            concept_mean_macro_f1_score_2 = np.mean(concept_macro_f1_scores_2)

    print(f"Epoch {epoch + 1}: Same Ratio = {same_ratio*100/num_concept_labels}")
    # print(f"Epoch {epoch + 1}: Val concept Acc = {concept_val_accuracy*100/num_concept_labels} Val concept Macro F1 = {concept_mean_macro_f1_score*100}")
    print(f"Epoch {epoch + 1}: Val concept Acc 2 = {concept_val_accuracy_2*100/num_concept_labels} Val concept Macro F1 2 = {concept_mean_macro_f1_score_2*100}")
    print(f"Epoch {epoch + 1}: Val Acc = {val_accuracy*100} Val Macro F1 = {mean_macro_f1_score*100}")
    print(f"Epoch {epoch + 1}: Val Acc 2 = {val_accuracy2*100} Val Macro F1 2=  {mean_macro_f1_score_2*100}")
    if val_accuracy > best_acc_score:
        best_acc_score = val_accuracy
        torch.save(model, "./"+model_name+"_joint.pth")
        torch.save(ModelXtoCtoY_layer, "./"+model_name+"_ModelXtoCtoY_layer_joint.pth")

####################### test
num_epochs = 1
print("Test!")
model = torch.load("./"+model_name+"_joint.pth")
ModelXtoCtoY_layer = torch.load("./"+model_name+"_ModelXtoCtoY_layer_joint.pth") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            decoded_sentences = [tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in input_ids]
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            food_concept = batch["food_concept"].to(device)
            ambiance_concept=batch["ambiance_concept"].to(device)
            service_concept=batch["service_concept"].to(device)
            noise_concept=batch["noise_concept"].to(device)
            if data_type != "pure_cebab":
                cleanliness_concept = batch["cleanliness_concept"].to(device)
                price_concept = batch["price_concept"].to(device)
                location_concept = batch["location_concept"].to(device)
                menu_variety_concept = batch["menu_variety_concept"].to(device)
                waiting_time_concept = batch["waiting_time_concept"].to(device)
                waiting_area_concept = batch["waiting_area_concept"].to(device)        
            concept_labels=batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels)
            concept_labels = concept_labels.contiguous().view(-1)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if model_name == 'lstm':
                pooled_output = outputs
            else:
                pooled_output = outputs.last_hidden_state.mean(1)  
            outputs,y_pred_2, c_emb_2, c_pred_2 = ModelXtoCtoY_layer(pooled_output)  

            XtoC_output = outputs [1:] 
            XtoY_output = outputs [0:1]         
            predictions = torch.argmax(XtoY_output[0], axis=1)
            test_accuracy += torch.sum(predictions == label).item()
            # ====== pred_2
            y_pred_2_res = torch.argmax(y_pred_2, axis=1)
            test_accuracy2 += torch.sum(y_pred_2_res == label).item()
            predict_labels_2 = np.append(predict_labels_2, y_pred_2_res.cpu().numpy())
            # ========
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())
            #concept accuracy
            # XtoC_logits = torch.cat(XtoC_output, dim=0)
            # concept_predictions = torch.argmax(XtoC_logits, axis=1)

            # concept_test_accuracy += torch.sum(concept_predictions == concept_labels).item()
            # concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())
            # concept_predictions = concept_predictions.reshape(-1,num_concept_labels)  # reshape 二维向量[batch_size*num_concept_labels]
        

            # 用 concept2 =========
            concept_predictions_2 = categorize_tensor(c_pred_2).transpose(0, 1).contiguous().view(-1, 1).squeeze(1)
            concept_test_accuracy_2 += torch.sum(concept_predictions_2 == concept_labels).item()
                # 看两个是否 sanme
            # same_ratio+=torch.sum(concept_predictions_2 ==concept_predictions).item()
                # =======
            concept_predict_labels_2 = np.append(concept_predict_labels_2, concept_predictions_2.cpu().numpy())
            # ==========
            #=====

            import logging
            # 设置日志配置
            logging.basicConfig(
                filename='output.log',  # 日志文件名称
                level=logging.DEBUG,    # 设置日志级别
                format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
            )

            # 将信息写入日志的函数
            def log_output():
                logging.debug('-----concept_predictions_2:')
                logging.debug(categorize_tensor(c_pred_2))  # 假设 categorize_tensor 是你的函数
                logging.debug(c_pred_2)

                logging.debug('-----concept:')
                logging.debug(concept_labels.reshape(10, concept_labels.numel()//10).T)

                logging.debug('-----y_pred_2:')
                logging.debug(y_pred_2_res)

                logging.debug('-----sentsences:')
                logging.debug(decoded_sentences)

                logging.debug('-----true label:')
                logging.debug(label)

                # 假设 ModelXtoCtoY_layer 是你的模型对象
                logging.debug(ModelXtoCtoY_layer.explain(c_emb_2, c_pred_2))

            # 调用 log_output 函数来写日志
            # log_output()
           
             #=====
        test_accuracy /= len(test_dataset)
        test_accuracy2 /= len(test_dataset)
        num_labels = len(np.unique(true_labels))

        #concept_test_accuracy /= len(test_dataset)
        concept_test_accuracy_2 /= len(val_dataset)
        concept_num_true_labels = len(np.unique(concept_true_labels))
        
        macro_f1_scores = []
        for label in range(num_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)

        macro_f1_scores_2 = []
        for label in range(num_labels):
            label_pred = np.array(predict_labels_2) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores_2.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score_2 = np.mean(macro_f1_scores_2)

        # concept_macro_f1_scores = []
        # for concept_label in range(concept_num_true_labels):
        #     concept_label_pred = np.array(concept_predict_labels) == concept_label
        #     concept_label_true = np.array(concept_true_labels) == concept_label
        #     concept_macro_f1_scores.append(f1_score(concept_label_true, concept_label_pred, average='macro'))
        #     concept_mean_macro_f1_score = np.mean(concept_macro_f1_scores)

        concept_macro_f1_scores_2 = []
        for concept_label in range(concept_num_true_labels):
            concept_label_pred_2 = np.array(concept_predict_labels_2) == concept_label
            concept_label_true = np.array(concept_true_labels) == concept_label
            concept_macro_f1_scores_2.append(f1_score(concept_label_true, concept_label_pred_2, average='macro'))
            concept_mean_macro_f1_score_2 = np.mean(concept_macro_f1_scores_2)

    print(ModelXtoCtoY_layer.explain(c_emb_2, c_pred_2))
    #print(f"Epoch {epoch + 1}: Test concept Acc = {concept_test_accuracy*100/num_concept_labels} Test concept Macro F1 = {concept_mean_macro_f1_score*100}")
    print(f"Epoch {epoch + 1}: Test concept Acc 2 = {concept_test_accuracy_2*100/num_concept_labels} Test concept Macro F1 2 = {concept_mean_macro_f1_score_2*100}")
    print(f"Epoch {epoch + 1}: Test Acc = {test_accuracy*100} Test Macro F1 = {mean_macro_f1_score*100}")
    print(f"Epoch {epoch + 1}: Test Acc 2= {test_accuracy2*100} Test Macro F1 2 =  {mean_macro_f1_score_2*100}")

    print(f"{concept_loss_weight}\t"
      f"{y2_weight}\t"
      f"{num_epochs_train}\t"
      f"{concept_test_accuracy_2 * 100 / num_concept_labels:.4f}\t"
      f"{concept_mean_macro_f1_score_2 * 100:.4f}\t"
      f"{test_accuracy * 100:.4f}\t"
      f"{mean_macro_f1_score * 100:.4f}\t"
      f"{test_accuracy2 * 100:.4f}\t"
      f"{mean_macro_f1_score_2 * 100:.4f}")

    print(loss_list)