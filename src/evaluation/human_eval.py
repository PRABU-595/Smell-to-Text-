"""
Human evaluation framework for smell-to-molecule predictions
"""
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


class HumanEvaluationStudy:
    """Framework for human evaluation of model predictions."""
    
    def __init__(self, study_name: str, output_dir: str = "outputs/human_eval"):
        self.study_name = study_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples = []
        self.evaluators = []
        self.responses = []
    
    def add_sample(self, sample_id: str, description: str,
                   predicted_chemicals: List[Dict], ground_truth: Optional[List[Dict]] = None):
        self.samples.append({
            'id': sample_id, 'description': description,
            'predictions': predicted_chemicals, 'ground_truth': ground_truth
        })
    
    def register_evaluator(self, evaluator_id: str, name: str, expertise: str = 'layperson'):
        self.evaluators.append({
            'id': evaluator_id, 'name': name, 'expertise': expertise,
            'registered_at': datetime.now().isoformat()
        })
    
    def create_evaluation_form(self, evaluator_id: str, num_samples: Optional[int] = None) -> List[Dict]:
        import random
        samples = self.samples.copy()
        random.shuffle(samples)
        if num_samples:
            samples = samples[:num_samples]
        
        form = [{'sample_id': s['id'], 'description': s['description'],
                 'predictions': s['predictions'], 'questions': self._get_questions()} for s in samples]
        
        with open(self.output_dir / f"form_{evaluator_id}.json", 'w') as f:
            json.dump(form, f, indent=2)
        return form
    
    def _get_questions(self) -> List[Dict]:
        return [
            {'id': 'q1_relevance', 'question': 'How relevant are the predictions?', 'type': 'scale', 'min': 1, 'max': 5},
            {'id': 'q2_accuracy', 'question': 'Would these chemicals produce this smell?', 'type': 'scale', 'min': 1, 'max': 5},
            {'id': 'q3_usefulness', 'question': 'Is this prediction useful?', 'type': 'scale', 'min': 1, 'max': 5},
            {'id': 'q4_comments', 'question': 'Additional comments?', 'type': 'text'}
        ]
    
    def submit_response(self, evaluator_id: str, sample_id: str, answers: Dict):
        self.responses.append({'evaluator_id': evaluator_id, 'sample_id': sample_id,
                               'answers': answers, 'submitted_at': datetime.now().isoformat()})
        with open(self.output_dir / f"{self.study_name}_responses.json", 'w') as f:
            json.dump(self.responses, f, indent=2)
    
    def compute_statistics(self) -> Dict:
        if not self.responses:
            return {}
        records = [{'evaluator_id': r['evaluator_id'], 'sample_id': r['sample_id'], **r['answers']} for r in self.responses]
        df = pd.DataFrame(records)
        stats = {}
        for col in ['q1_relevance', 'q2_accuracy', 'q3_usefulness']:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce')
                stats[col] = {'mean': float(values.mean()), 'std': float(values.std())}
        return stats
    
    def generate_report(self) -> str:
        stats = self.compute_statistics()
        report = f"# Human Evaluation Report: {self.study_name}\n\n"
        report += f"Samples: {len(self.samples)}, Evaluators: {len(self.evaluators)}, Responses: {len(self.responses)}\n\n"
        for k, v in stats.items():
            report += f"**{k}**: mean={v['mean']:.2f}, std={v['std']:.2f}\n"
        with open(self.output_dir / f"{self.study_name}_report.md", 'w') as f:
            f.write(report)
        return report
