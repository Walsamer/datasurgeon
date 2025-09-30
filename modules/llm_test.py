"""
Testing Framework for Kaggle Business/Analytics Datasets
Tests chatbot performance on real-world data scenarios
"""

import pandas as pd
from llm_chat import ChatModule
import time
from datetime import datetime

class KaggleDatasetTester:
    """Test chatbot with Kaggle business datasets"""
    def __init__(self):
        self.chat = ChatModule()
        self.test_results = []
        self.dataset_info = {
            'crocodile': {
                'name': 'Crocodile Species Dataset',
                'file': 'crocodile_data.csv',
                'expected_columns': ['species', 'length', 'weight', 'age', 'location'],
                'key_insights': [
                    'Species distribution',
                    'Size correlations',
                    'Geographic patterns'
                ]
            },
            'bi_cleaning': {
                'name': 'BI Data Cleaning Dataset',
                'file': 'bi_data_cleaning.csv',
                'expected_issues': [
                    'Missing values',
                    'Duplicate records',
                    'Inconsistent formatting',
                    'Outliers'
                ]
            },
            'employee': {
                'name': 'Employee Productivity Dataset',
                'file': 'employee_productivity.csv',
                'expected_columns': ['employee_id', 'department', 'productivity_score', 
                                    'hours_worked', 'training_hours'],
                'key_metrics': [
                    'Productivity by department',
                    'Training impact',
                    'Work hours correlation'
                ]
            },
            'fraud': {
                'name': 'SecurePay Credit Card Fraud Dataset',
                'file': 'credit_card_fraud.csv',
                'expected_columns': ['transaction_id', 'amount', 'merchant', 'is_fraud'],
                'key_patterns': [
                    'Fraud rate',
                    'High-risk transactions',
                    'Merchant patterns'
                ]
            }
        }
    
    def run_test_query(self, query, test_name, validate_fn=None):
        """Execute a single test query"""
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"Query: {query}")
        print("-"*60)
        
        start = time.time()
        response = self.chat.chat(query)
        duration = time.time() - start
        
        # Show response
        print(f"Response ({duration:.2f}s):")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        # Validate if function provided
        passed = True
        if validate_fn:
            try:
                passed = validate_fn(response)
                print(f"\nValidation: {'PASSED' if passed else 'FAILED'}")
            except Exception as e:
                print(f"\nValidation Error: {e}")
                passed = False
        
        # Store result
        self.test_results.append({
            'test_name': test_name,
            'query': query,
            'duration': duration,
            'passed': passed,
            'timestamp': datetime.now().isoformat()
        })
        
        return response, passed
    
    def test_crocodile_dataset(self, df):
        """Test with Crocodile Species dataset"""
        print("\n" + "="*70)
        print("TESTING: CROCODILE SPECIES DATASET")
        print("="*70)
        
        self.chat.set_dataframe(df)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        tests = [
            # Basic understanding
            ("What type of data is this?", 
             "Dataset Understanding",
             lambda r: "crocodile" in r.lower() or "species" in r.lower()),
            
            # Data quality
            ("Check for data quality issues",
             "Data Quality Assessment",
             lambda r: "missing" in r.lower() or "null" in r.lower()),
            
            # Statistical analysis
            ("What's the average size of crocodiles?",
             "Statistical Query",
             lambda r: any(word in r.lower() for word in ["length", "weight", "size"])),
            
            # Comparative analysis
            ("Compare different species",
             "Species Comparison",
             lambda r: "species" in r.lower()),
            
            # Pattern detection
            ("Any interesting patterns in the data?",
             "Pattern Recognition",
             None)
        ]
        
        for query, name, validator in tests:
            self.run_test_query(query, f"Crocodile: {name}", validator)
    
    def test_bi_cleaning_dataset(self, df):
        """Test with BI Data Cleaning dataset"""
        print("\n" + "="*70)
        print("TESTING: BI DATA CLEANING DATASET")
        print("="*70)
        
        self.chat.set_dataframe(df)
        print(f"Dataset shape: {df.shape}")
        
        tests = [
            # Data quality issues
            ("Identify all data quality problems",
             "Quality Issues Detection",
             lambda r: any(issue in r.lower() for issue in 
                          ["missing", "duplicate", "inconsistent", "outlier"])),
            
            # Missing data analysis
            ("How much data is missing and where?",
             "Missing Data Analysis",
             lambda r: "missing" in r.lower() or "null" in r.lower()),
            
            # Cleaning recommendations
            ("How should I clean this dataset?",
             "Cleaning Recommendations",
             lambda r: any(action in r.lower() for action in 
                          ["remove", "fill", "impute", "drop", "clean"])),
            
            # Data validation
            ("Are there any suspicious values?",
             "Anomaly Detection",
             None),
            
            # Completeness check
            ("What percentage of the data is complete?",
             "Completeness Assessment",
             lambda r: "%" in r or "percent" in r.lower())
        ]
        
        for query, name, validator in tests:
            self.run_test_query(query, f"BI Cleaning: {name}", validator)
    
    def test_employee_productivity_dataset(self, df):
        """Test with Employee Productivity dataset"""
        print("\n" + "="*70)
        print("TESTING: EMPLOYEE PRODUCTIVITY DATASET")
        print("="*70)
        
        self.chat.set_dataframe(df)
        print(f"Dataset shape: {df.shape}")
        
        tests = [
            # Overview
            ("Summarize employee productivity",
             "Productivity Overview",
             lambda r: "productivity" in r.lower()),
            
            # Department analysis
            ("Which department is most productive?",
             "Department Comparison",
             lambda r: "department" in r.lower()),
            
            # Correlation analysis
            ("Does working more hours increase productivity?",
             "Hours vs Productivity",
             lambda r: "hour" in r.lower()),
            
            # Training impact
            ("Is training effective?",
             "Training Effectiveness",
             lambda r: "training" in r.lower()),
            
            # Recommendations
            ("How can we improve productivity?",
             "Improvement Recommendations",
             None),
            
            # Outlier detection
            ("Find exceptional performers",
             "High Performer Detection",
             lambda r: any(word in r.lower() for word in 
                          ["high", "top", "best", "exceptional"]))
        ]
        
        for query, name, validator in tests:
            self.run_test_query(query, f"Employee: {name}", validator)
    
    def test_fraud_dataset(self, df):
        """Test with Credit Card Fraud dataset"""
        print("\n" + "="*70)
        print("TESTING: CREDIT CARD FRAUD DATASET")
        print("="*70)
        
        self.chat.set_dataframe(df)
        print(f"Dataset shape: {df.shape}")
        
        tests = [
            # Fraud rate
            ("What's the fraud rate?",
             "Fraud Rate Calculation",
             lambda r: "%" in r or "percent" in r.lower() or "rate" in r.lower()),
            
            # Pattern analysis
            ("What are common fraud patterns?",
             "Fraud Pattern Detection",
             lambda r: "fraud" in r.lower()),
            
            # Risk factors
            ("What makes a transaction high risk?",
             "Risk Factor Analysis",
             lambda r: "risk" in r.lower() or "amount" in r.lower()),
            
            # Merchant analysis
            ("Which merchants have most fraud?",
             "Merchant Risk Analysis",
             lambda r: "merchant" in r.lower()),
            
            # Model suggestions
            ("How would you detect fraud?",
             "Detection Strategy",
             None),
            
            # Data balance
            ("Is the data balanced?",
             "Class Balance Check",
             lambda r: any(word in r.lower() for word in 
                          ["balanced", "imbalanced", "ratio", "class"]))
        ]
        
        for query, name, validator in tests:
            self.run_test_query(query, f"Fraud: {name}", validator)
    
    def generate_summary_report(self):
        """Generate test summary report"""
        print("\n" + "="*70)
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*70)
        
        if not self.test_results:
            print("No tests run yet!")
            return
        
        # Overall metrics
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['passed'])
        avg_time = sum(r['duration'] for r in self.test_results) / total
        
        print(f"\nOverall Results:")
        print(f"  Total Tests: {total}")
        print(f"  Passed: {passed}/{total} ({100*passed/total:.1f}%)")
        print(f"  Average Response Time: {avg_time:.2f}s")
        
        # By dataset breakdown
        datasets = ['Crocodile', 'BI Cleaning', 'Employee', 'Fraud']
        for dataset in datasets:
            dataset_tests = [r for r in self.test_results 
                            if dataset in r['test_name']]
            if dataset_tests:
                dataset_passed = sum(1 for r in dataset_tests if r['passed'])
                print(f"\n{dataset} Dataset:")
                print(f"  Tests: {len(dataset_tests)}")
                print(f"  Pass Rate: {100*dataset_passed/len(dataset_tests):.1f}%")
                print(f"  Avg Time: {sum(r['duration'] for r in dataset_tests)/len(dataset_tests):.2f}s")
        
        # Failed tests
        failed = [r for r in self.test_results if not r['passed']]
        if failed:
            print(f"\nFailed Tests ({len(failed)}):")
            for r in failed:
                print(f"  - {r['test_name']}: {r['query'][:50]}...")
        
        # Save to CSV
        results_df = pd.DataFrame(self.test_results)
        filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nDetailed results saved to '{filename}'")


# Test query templates for manual testing
BUSINESS_QUERIES = {
    'exploratory': [
        "What does this data represent?",
        "Give me a quick overview",
        "What should I look at first?",
        "What's the most interesting finding?"
    ],
    'quality': [
        "Check data quality",
        "Find missing values",
        "Are there duplicates?",
        "Any data inconsistencies?",
        "Is this data reliable?"
    ],
    'statistical': [
        "Calculate basic statistics",
        "What's the distribution?",
        "Find correlations",
        "Identify outliers",
        "Show me the ranges"
    ],
    'analytical': [
        "What patterns do you see?",
        "What's driving the results?",
        "Find the key factors",
        "What's unusual here?",
        "Explain the trends"
    ],
    'actionable': [
        "What should we do next?",
        "How can we improve?",
        "What's the priority?",
        "Make recommendations",
        "What analysis is needed?"
    ]
}

def load_and_test_dataset(file_path, dataset_type, tester):
    """Helper function to load and test a dataset"""
    try:
        # Try different encodings if needed
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        print(f"\nLoaded {file_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()[:5]}..." if len(df.columns) > 5 
              else f"   Columns: {df.columns.tolist()}")
        

        if dataset_type == 'crocodile':
            tester.test_crocodile_dataset(df)
        elif dataset_type == 'bi_cleaning':
            tester.test_bi_cleaning_dataset(df)
        elif dataset_type == 'employee':
            tester.test_employee_productivity_dataset(df)
        elif dataset_type == 'fraud':
            tester.test_fraud_dataset(df)
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print(f"   Please download from Kaggle and save as '{file_path}'")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")


if __name__ == "__main__":
    print("="*70)
    print("KAGGLE DATASET TESTING FRAMEWORK")
    print("="*70)
    
    tester = KaggleDatasetTester()
    
    # datasetpaths)
    # datasets = [
    #     ([insert file path here] data/other_testsets/data/crocodile_data.csv', 'crocodile'),
    #     ([insert file path here] data/other_testsets/bi.csv', 'bi_cleaning'),
    #     ([insert file path here] data/other_testsets/employee_productivity.csv', 'employee'),
    #     ([insert file path here] data/other_testsets/creditcard.csv', 'fraud')
    # ]
    
    for file_path, dataset_type in datasets:
        load_and_test_dataset(file_path, dataset_type, tester)
    
    tester.generate_summary_report()