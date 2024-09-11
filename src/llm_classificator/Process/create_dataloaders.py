from transformers import XLNetTokenizer
from torch.utils.data import DataLoader
from modules.questions_dataset import QuestionsDataset
from typing import Any, Dict

def create_dataloaders(data: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Crea un DataLoader para el entrenamiento y otro para la evaluación a partir de los datos proporcionados.
    
    Args:
        data (Dict[str, Any]): Diccionario con los datos de entrenamiento y evaluación.
        
    Returns:
        DataLoader: DataLoader para el entrenamiento.
        DataLoader: DataLoader para la evaluación.
    '''

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    # Reutilizar la clase QuestionsDataset y el tokenizer
    train_dataset = QuestionsDataset(data['train_questions'], data['train_labels'], tokenizer, max_len=32)
    eval_dataset = QuestionsDataset(data['eval_questions'], data['eval_labels'], tokenizer, max_len=32)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
    
    return {'train_dataloader': train_dataloader, 'eval_dataloader': eval_dataloader}