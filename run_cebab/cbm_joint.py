import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import (
    RobertaTokenizer, RobertaModel,
    BertModel, BertTokenizer,
    GPT2Model, GPT2Tokenizer
)
from gensim.models import FastText
from datasets import load_dataset

# Custom modules (assumed to be in local directory)
from cbm_models import Model_ConceptReasoning
from tools import categorize_tensor

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
mode = 'joint'
model_name = 'lstm'  # 'bert-base-uncased' / 'roberta-base' / 'gpt2' / 'lstm'
data_type = "aug_cebab_yelp"  # "pure_cebab"/"aug_cebab"/"aug_yelp"/"aug_cebab_yelp"

# Paths
base_path = '/mnt/llm_test_afs/code/yyb/kejieshi/models'
fasttext_path = os.path.join(base_path, 'fasttext/cc.en.300.bin')
bert_path = os.path.join(base_path, 'bert-base-uncased')
roberta_path = os.path.join(base_path, 'roberta-base')
gpt2_path = os.path.join(base_path, 'gpt2')

# Training params
max_len = 512
batch_size = 8
lambda_XtoC = 0.5
is_aux_logits = False
num_labels = 5
num_epochs = 25
num_epochs_train = 25
learning_rate = 1e-2
num_each_concept_classes = 3
concept_loss_weight = 100
y2_weight = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Model Classes & Tokenizer Setup
# ==========================================

class BiLSTMWithDotAttention(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, fasttext_model):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        embeddings = fasttext_model.wv.vectors
        self.embedding.weight = torch.nn.Parameter(torch.tensor(embeddings))
        self.embedding.weight.requires_grad = False
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )

    def forward(self, input_ids, attention_mask):
        # input_lengths = attention_mask.sum(dim=1) # Unused in original code, but kept logic flow
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)
        weights = F.softmax(torch.bmm(output, output.transpose(1, 2)), dim=2)
        attention = torch.bmm(weights, output)
        logits = self.classifier(attention.mean(1))
        return logits

# Load Tokenizer and Backbone Model
if model_name == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
    model = RobertaModel.from_pretrained(roberta_path)
elif model_name == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertModel.from_pretrained(bert_path)
elif model_name == 'gpt2':
    model = GPT2Model.from_pretrained(gpt2_path)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path)
    tokenizer.pad_token = tokenizer.eos_token
elif model_name == 'lstm':
    fasttext_model = FastText.load_fasttext_format(fasttext_path)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BiLSTMWithDotAttention(len(tokenizer.vocab), 300, 128, fasttext_model)

# ==========================================
# 3. Data Preparation
# ==========================================

CEBaB = {}
num_concept_labels = 4 # Default

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
    CEBaB[train_split] = pd.read_csv("../dataset/cebab/train_cebab_new_concept_single.csv")
    CEBaB[test_split] = pd.read_csv("../dataset/cebab/test_cebab_new_concept_single.csv")
    CEBaB[val_split] = pd.read_csv("../dataset/cebab/dev_cebab_new_concept_single.csv")
elif data_type == "aug_yelp":
    num_concept_labels = 10
    train_split = "train_aug_yelp"
    test_split = "test_aug_yelp"
    val_split = "val_aug_yelp"
    CEBaB[train_split] = pd.read_csv("../dataset/cebab/train_yelp_exclusive_new_concept_single.csv")
    CEBaB[test_split] = pd.read_csv("../dataset/cebab/test_yelp_new_concept_single.csv")
    CEBaB[val_split] = pd.read_csv("../dataset/cebab/dev_yelp_new_concept_single.csv")
elif data_type == "aug_cebab_yelp":
    num_concept_labels = 10
    train_split = "train_aug_cebab_yelp"
    test_split = "test_aug_cebab_yelp"
    val_split = "val_aug_cebab_yelp"
    
    # Load separate files and concat
    CEBaB[train_split] = pd.concat([
        pd.read_csv("../dataset/cebab/train_cebab_new_concept_single.csv"),
        pd.read_csv("../dataset/cebab/train_yelp_exclusive_new_concept_single.csv")
    ], ignore_index=True)
    
    CEBaB[test_split] = pd.concat([
        pd.read_csv("../dataset/cebab/test_cebab_new_concept_single.csv"),
        pd.read_csv("../dataset/cebab/test_yelp_new_concept_single.csv")
    ], ignore_index=True)
    
    CEBaB[val_split] = pd.concat([
        pd.read_csv("../dataset/cebab/dev_cebab_new_concept_single.csv"),
        pd.read_csv("../dataset/cebab/dev_yelp_new_concept_single.csv")
    ], ignore_index=True)

class MyDataset(Dataset):
    def __init__(self, split, skip_class="no majority"):
        self.data = CEBaB[split]
        self.labels = self.data["review_majority"]
        self.text = self.data["description"]
       
        self.food_aspect = self.data["food_aspect_majority"]
        self.ambiance_aspect = self.data["ambiance_aspect_majority"]
        self.service_aspect = self.data["service_aspect_majority"]
        self.noise_aspect = self.data["noise_aspect_majority"]

        if data_type != "pure_cebab":
            self.cleanliness_aspect = self.data["cleanliness"]
            self.price_aspect = self.data["price"]
            self.location_aspect = self.data["location"]
            self.menu_variety_aspect = self.data["menu variety"]
            self.waiting_time_aspect = self.data["waiting time"]
            self.waiting_area_aspect = self.data["waiting area"]

        self.map_dict = {"Negative": 0, "Positive": 2, "unknown": 1, "": 1, "no majority": 1}

        self.skip_class = skip_class
        if skip_class is not None:
            self.indices = [i for i, label in enumerate(self.labels) if label != skip_class]
        else:
            self.indices = range(len(self.labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]
        text = self.text[idx]
        label = int(self.labels[idx]) - 1

        # Gold labels
        food_concept = self.map_dict[self.food_aspect[idx]]
        ambiance_concept = self.map_dict[self.ambiance_aspect[idx]]
        service_concept = self.map_dict[self.service_aspect[idx]]
        noise_concept = self.map_dict[self.noise_aspect[idx]]
        
        # Extended concepts for non-pure datasets
        cleanliness_concept = price_concept = location_concept = 0
        menu_variety_concept = waiting_time_concept = waiting_area_concept = 0

        if data_type != "pure_cebab":
            cleanliness_concept = self.map_dict[self.cleanliness_aspect[idx]]
            price_concept = self.map_dict[self.price_aspect[idx]]
            location_concept = self.map_dict[self.location_aspect[idx]]
            menu_variety_concept = self.map_dict[self.menu_variety_aspect[idx]]
            waiting_time_concept = self.map_dict[self.waiting_time_aspect[idx]]
            waiting_area_concept = self.map_dict[self.waiting_area_aspect[idx]]
            
            concept_labels = [
                food_concept, ambiance_concept, service_concept, noise_concept,
                cleanliness_concept, price_concept, location_concept,
                menu_variety_concept, waiting_time_concept, waiting_area_concept
            ]
        else: 
            concept_labels = [food_concept, ambiance_concept, service_concept, noise_concept]

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
            "food_concept": torch.tensor(food_concept, dtype=torch.long),
            "ambiance_concept": torch.tensor(ambiance_concept, dtype=torch.long),
            "service_concept": torch.tensor(service_concept, dtype=torch.long),
            "noise_concept": torch.tensor(noise_concept, dtype=torch.long),
            "concept_labels": torch.tensor(concept_labels, dtype=torch.long)
        }
        
        if data_type != "pure_cebab":
            result.update({
                "cleanliness_concept": torch.tensor(cleanliness_concept, dtype=torch.long),
                "price_concept": torch.tensor(price_concept, dtype=torch.long),
                "location_concept": torch.tensor(location_concept, dtype=torch.long),
                "menu_variety_concept": torch.tensor(menu_variety_concept, dtype=torch.long),
                "waiting_time_concept": torch.tensor(waiting_time_concept, dtype=torch.long),
                "waiting_area_concept": torch.tensor(waiting_area_concept, dtype=torch.long),
            })
            
        return result

# Data Loaders
train_dataset = MyDataset(train_split)
test_dataset = MyDataset(test_split)
val_dataset = MyDataset(val_split)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ==========================================
# 4. Model Initialization
# ==========================================

if model_name == 'lstm':
    ModelXtoCtoY_layer = Model_ConceptReasoning(
        concept_classes=num_each_concept_classes, 
        label_classes=num_labels, 
        n_attributes=num_concept_labels, 
        bottleneck=True, expand_dim=0, 
        n_class_attr=num_each_concept_classes, 
        use_relu=False, use_sigmoid=False, 
        Lstm=True, aux_logits=is_aux_logits
    )
else:
    ModelXtoCtoY_layer = Model_ConceptReasoning(
        concept_classes=num_each_concept_classes, 
        label_classes=num_labels, 
        n_attributes=num_concept_labels, 
        bottleneck=True, expand_dim=0, 
        n_class_attr=num_each_concept_classes, 
        use_relu=False, use_sigmoid=False, 
        aux_logits=is_aux_logits
    )

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(ModelXtoCtoY_layer.parameters()), 
    lr=learning_rate
)

if model_name == 'lstm':
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

loss_fn = torch.nn.CrossEntropyLoss()
ModelXtoCtoY_layer.to(device)
model.to(device)

# ==========================================
# 5. Training Loop
# ==========================================

loss_list = []
best_acc_score = 0

for epoch in range(num_epochs):
    # --- Train ---
    ModelXtoCtoY_layer.train()
    model.train()
    single_loss_list = []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        concept_labels = batch["concept_labels"].to(device)
        concept_labels = torch.t(concept_labels)
        concept_labels = concept_labels.contiguous().view(-1) 

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        if model_name == 'lstm':
            pooled_output = outputs
        else:
            pooled_output = outputs.last_hidden_state.mean(1)  
            
        outputs, y_pred_2, c_emb_2, c_pred_2 = ModelXtoCtoY_layer(pooled_output)  
        XtoC_output = outputs[1:] 
        XtoY_output = outputs[0:1]

        # Losses
        XtoY_loss = loss_fn(XtoY_output[0], label)
        XtoY_loss_2 = loss_fn(y_pred_2, label)

        mapped_labels = concept_labels.float() * 0.5
        c_pred_2_transposed_flat = c_pred_2.transpose(0, 1).contiguous().view(-1, 1)
        concept_loss = F.mse_loss(c_pred_2_transposed_flat.squeeze(1), mapped_labels)
        
        loss = XtoY_loss + (XtoY_loss_2 * y2_weight) + (concept_loss * concept_loss_weight)
        
        single_loss_list.append(concept_loss.item())
        loss.backward()
        optimizer.step()
    
    loss_list.append(np.mean(single_loss_list))

    # --- Validation ---
    model.eval()
    ModelXtoCtoY_layer.eval()
    
    val_accuracy = 0.
    val_accuracy2 = 0.
    concept_val_accuracy_2 = 0.
    same_ratio = 0.
    
    predict_labels = np.array([])
    predict_labels_2 = np.array([])
    true_labels = np.array([])
    concept_predict_labels_2 = np.array([])
    concept_true_labels = np.array([])

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            concept_labels = batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels)
            concept_labels = concept_labels.contiguous().view(-1)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if model_name == 'lstm':
                pooled_output = outputs
            else:
                pooled_output = outputs.last_hidden_state.mean(1)  
                
            outputs, y_pred_2, c_emb_2, c_pred_2 = ModelXtoCtoY_layer(pooled_output)  
            XtoY_output = outputs[0:1]         
            
            # Predictions
            predictions = torch.argmax(XtoY_output[0], axis=1)
            val_accuracy += torch.sum(predictions == label).item()

            y_pred_2_res = torch.argmax(y_pred_2, axis=1)
            val_accuracy2 += torch.sum(y_pred_2_res == label).item()
            predict_labels_2 = np.append(predict_labels_2, y_pred_2_res.cpu().numpy())

            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())

            # Concept Accuracy
            concept_predictions_2 = categorize_tensor(c_pred_2).transpose(0, 1).contiguous().view(-1, 1).squeeze(1)
            concept_val_accuracy_2 += torch.sum(concept_predictions_2 == concept_labels).item()
            
            concept_predict_labels_2 = np.append(concept_predict_labels_2, concept_predictions_2.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())

    # Metrics Calculation
    val_accuracy /= len(val_dataset)
    val_accuracy2 /= len(val_dataset)
    num_unique_labels = len(np.unique(true_labels))
    concept_val_accuracy_2 /= len(val_dataset)
    same_ratio /= len(val_dataset)
    concept_num_true_labels = len(np.unique(concept_true_labels))

    macro_f1_scores = []
    for l in range(num_unique_labels):
        label_pred = np.array(predict_labels) == l
        label_true = np.array(true_labels) == l
        macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
    mean_macro_f1_score = np.mean(macro_f1_scores)

    macro_f1_scores_2 = []
    for l in range(num_unique_labels):
        label_pred = np.array(predict_labels_2) == l
        label_true = np.array(true_labels) == l
        macro_f1_scores_2.append(f1_score(label_true, label_pred, average='macro'))
    mean_macro_f1_score_2 = np.mean(macro_f1_scores_2)

    concept_macro_f1_scores_2 = []
    for cl in range(concept_num_true_labels):
        concept_label_pred_2 = np.array(concept_predict_labels_2) == cl
        concept_label_true = np.array(concept_true_labels) == cl
        concept_macro_f1_scores_2.append(f1_score(concept_label_true, concept_label_pred_2, average='macro'))
    concept_mean_macro_f1_score_2 = np.mean(concept_macro_f1_scores_2)

    print(f"Epoch {epoch + 1}: Same Ratio = {same_ratio*100/num_concept_labels}")
    print(f"Epoch {epoch + 1}: Val concept Acc 2 = {concept_val_accuracy_2*100/num_concept_labels} Val concept Macro F1 2 = {concept_mean_macro_f1_score_2*100}")
    print(f"Epoch {epoch + 1}: Val Acc = {val_accuracy*100} Val Macro F1 = {mean_macro_f1_score*100}")
    print(f"Epoch {epoch + 1}: Val Acc 2 = {val_accuracy2*100} Val Macro F1 2 = {mean_macro_f1_score_2*100}")

    if val_accuracy > best_acc_score:
        best_acc_score = val_accuracy
        torch.save(model, f"./{model_name}_joint.pth")
        torch.save(ModelXtoCtoY_layer, f"./{model_name}_ModelXtoCtoY_layer_joint.pth")

# ==========================================
# 6. Testing Loop
# ==========================================
print("Test!")
num_epochs_test = 1
model = torch.load(f"./{model_name}_joint.pth")
ModelXtoCtoY_layer = torch.load(f"./{model_name}_ModelXtoCtoY_layer_joint.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs_test):
    test_accuracy = 0.
    test_accuracy2 = 0.
    concept_test_accuracy_2 = 0.
    
    predict_labels = np.array([])
    predict_labels_2 = np.array([])
    true_labels = np.array([])
    concept_predict_labels_2 = np.array([])
    concept_true_labels = np.array([])

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            concept_labels = batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels)
            concept_labels = concept_labels.contiguous().view(-1)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if model_name == 'lstm':
                pooled_output = outputs
            else:
                pooled_output = outputs.last_hidden_state.mean(1)  
                
            outputs, y_pred_2, c_emb_2, c_pred_2 = ModelXtoCtoY_layer(pooled_output)  
            XtoY_output = outputs[0:1]         
            
            predictions = torch.argmax(XtoY_output[0], axis=1)
            test_accuracy += torch.sum(predictions == label).item()

            y_pred_2_res = torch.argmax(y_pred_2, axis=1)
            test_accuracy2 += torch.sum(y_pred_2_res == label).item()
            predict_labels_2 = np.append(predict_labels_2, y_pred_2_res.cpu().numpy())

            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())
            
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())

            concept_predictions_2 = categorize_tensor(c_pred_2).transpose(0, 1).contiguous().view(-1, 1).squeeze(1)
            concept_test_accuracy_2 += torch.sum(concept_predictions_2 == concept_labels).item()
            concept_predict_labels_2 = np.append(concept_predict_labels_2, concept_predictions_2.cpu().numpy())

    # Final Metrics
    test_accuracy /= len(test_dataset)
    test_accuracy2 /= len(test_dataset)
    num_unique_labels = len(np.unique(true_labels))
    concept_test_accuracy_2 /= len(val_dataset) # Kept logic from original (divided by val_dataset?)
    concept_num_true_labels = len(np.unique(concept_true_labels))
    
    macro_f1_scores = []
    for l in range(num_unique_labels):
        label_pred = np.array(predict_labels) == l
        label_true = np.array(true_labels) == l
        macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
    mean_macro_f1_score = np.mean(macro_f1_scores)

    macro_f1_scores_2 = []
    for l in range(num_unique_labels):
        label_pred = np.array(predict_labels_2) == l
        label_true = np.array(true_labels) == l
        macro_f1_scores_2.append(f1_score(label_true, label_pred, average='macro'))
    mean_macro_f1_score_2 = np.mean(macro_f1_scores_2)

    concept_macro_f1_scores_2 = []
    for cl in range(concept_num_true_labels):
        concept_label_pred_2 = np.array(concept_predict_labels_2) == cl
        concept_label_true = np.array(concept_true_labels) == cl
        concept_macro_f1_scores_2.append(f1_score(concept_label_true, concept_label_pred_2, average='macro'))
    concept_mean_macro_f1_score_2 = np.mean(concept_macro_f1_scores_2)

    # Explanation and Final Output
    print(ModelXtoCtoY_layer.explain(c_emb_2, c_pred_2))
    print(f"Epoch {epoch + 1}: Test concept Acc 2 = {concept_test_accuracy_2*100/num_concept_labels} Test concept Macro F1 2 = {concept_mean_macro_f1_score_2*100}")
    print(f"Epoch {epoch + 1}: Test Acc = {test_accuracy*100} Test Macro F1 = {mean_macro_f1_score*100}")
    print(f"Epoch {epoch + 1}: Test Acc 2 = {test_accuracy2*100} Test Macro F1 2 = {mean_macro_f1_score_2*100}")

    print(f"{concept_loss_weight}\t"
          f"{y2_weight}\t"
          f"{num_epochs_train}\t"
          f"{concept_test_accuracy_2 * 100 / num_concept_labels:.4f}\t"
          f"{concept_mean_macro_f1_score_2 * 100:.4f}\t"
          f"{test_accuracy * 100:.4f}\t"
          f"{mean_macro_f1_score * 100:.4f}\t"
          f"{test_accuracy2 * 100:.4f}\t"
          f"{mean_macro_f1_score_2 * 100:.4f}")

    # Note: loss_list accumulates training losses across all training epochs, not test
    # print(loss_list) # Removed per instructions to remove extra prints, but can be uncommented if needed.