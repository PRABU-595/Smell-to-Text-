# API Reference

## Data Module

### SmellDataset
```python
from src.data.dataset import SmellDataset

dataset = SmellDataset(
    csv_path: str,           # Path to CSV file
    tokenizer: Tokenizer,    # HuggingFace tokenizer
    max_length: int = 128,   # Maximum sequence length
    num_chemicals: int = 300 # Number of chemical classes
)
```

### Scrapers
```python
from src.data.scrapers.fragrantica_scraper import FragranticaScraper
from src.data.scrapers.goodscents_scraper import GoodScentsScraper
from src.data.scrapers.pubchem_api import PubChemClient
```

## Models Module

### SmellToMoleculeModel
```python
from src.models.neobert_model import SmellToMoleculeModel

model = SmellToMoleculeModel(
    model_name: str = 'bert-base-uncased',
    num_chemicals: int = 300,
    dropout: float = 0.3
)
outputs, cls_embedding = model(input_ids, attention_mask)
```

### Baselines
```python
from src.models.tfidf_baseline import TFIDFBaseline
from src.models.rule_based import RuleBasedPredictor
```

## Training Module

### Trainer
```python
from src.training.trainer import Trainer

trainer = Trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict
)
trainer.train(epochs=30)
```

## Evaluation Module

### MetricsCalculator
```python
from src.evaluation.metrics import MetricsCalculator

calculator = MetricsCalculator()
results = calculator.compute_all_metrics(predictions, labels)
```
