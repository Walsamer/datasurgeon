"""
Tricky Real-World Scenarios that Often Break Chatbots
These tests simulate actual user behavior and common pitfalls
"""

import pandas as pd
import numpy as np
from llm_chat import ChatModule

class TrickyScenarioTests:
    """Tests based on real-world problematic scenarios"""
    
    def __init__(self):
        self.chat = ChatModule()
    
    def test_ambiguous_questions(self):
        """Test vague questions users actually ask"""
        # Create a sales dataset
        df = pd.DataFrame({
            'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
            'sales': [10000, 12000, 9000, 15000, 14000],
            'costs': [8000, 9000, 8500, 11000, 10000],
            'region': ['North', 'North', 'South', 'South', 'North']
        })
        self.chat.set_dataframe(df)
        
        vague_questions = [
            "is this good?",
            "what should I do?",
            "explain the numbers",
            "why?", 
            "tell me more",
            "and?",
            "so what?",
            "fix it",
            "make it better"
        ]
        
        print("\nTesting Ambiguous Questions:")
        print("-" * 50)
        
        for q in vague_questions:
            response = self.chat.chat(q)
            # Check if it asks for clarification or makes assumptions
            is_helpful = len(response) > 50  # At least tries to help
            print(f"Q: '{q}'")
            print(f"Response helpful: {'✓' if is_helpful else '✗'}")
            print(f"Response preview: {response[:100]}...")
            print()
    
    def test_column_confusion(self):
        """Test similar column names that confuse chatbots"""
        df = pd.DataFrame({
            'sales': [100, 200, 300],
            'Sales': [150, 250, 350],  # Different case
            'sale': [90, 190, 290],    # Similar name
            'sales_total': [400, 500, 600],
            'total_sales': [410, 510, 610],
            'sales2023': [300, 400, 500],
            'sales_2023': [310, 410, 510]
        })
        self.chat.set_dataframe(df)
        
        print("\nTesting Column Name Confusion:")
        print("-" * 50)
        
        # Ask about "sales" - which column will it use?
        response = self.chat.chat("What's the average sales?")
        print("Question: 'What's the average sales?'")
        print(f"Response: {response[:200]}")
        
        # The correct answer depends on which column it picks
        # 'sales': 200, 'Sales': 250, 'sale': 190, etc.
        
        # Check if it acknowledges the ambiguity
        mentions_multiple = any(word in response.lower() for word in 
                              ['multiple', 'several', 'which', 'different'])
        print(f"Acknowledges ambiguity: {'✓' if mentions_multiple else '✗'}")
    
    def test_data_type_assumptions(self):
        """Test wrong assumptions about data types"""
        df = pd.DataFrame({
            'id': ['001', '002', '003', '004', '005'],  # String IDs
            'score': [90, 85, 92, 88, 95],
            'grade': ['A', 'B', 'A', 'B', 'A'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],  # String dates
            'amount': ['$1,000', '$2,000', '$1,500', '$3,000', '$2,500']  # String amounts
        })
        self.chat.set_dataframe(df)
        
        print("\nTesting Data Type Assumptions:")
        print("-" * 50)
        
        tricky_questions = [
            ("What's the average ID?", "Should recognize IDs aren't numeric"),
            ("Calculate total amount", "Should handle currency strings"),
            ("What's the correlation between date and score?", "Should recognize date is string"),
            ("Add up all the grades", "Should recognize grades are categorical")
        ]
        
        for question, expected_behavior in tricky_questions:
            response = self.chat.chat(question)
            print(f"Q: {question}")
            print(f"Expected: {expected_behavior}")
            print(f"Response: {response[:150]}...")
            print()
    
    def test_missing_value_traps(self):
        """Test tricky missing value scenarios"""
        df = pd.DataFrame({
            'a': [1, 2, None, 4, 5],
            'b': [None, None, None, None, None],  # All missing
            'c': [1, 2, 3, 4, 5],  # No missing
            'd': [0, 0, np.nan, 0, 0],  # Zeros and NaN mixed
            'e': ['', '', 'value', '', '']  # Empty strings (not NaN)
        })
        self.chat.set_dataframe(df)
        
        print("\nTesting Missing Value Traps:")
        print("-" * 50)
        
        questions = [
            "What's the average of column b?",  # All NaN
            "Is column e empty?",  # Empty strings vs NaN
            "How many zeros in column d?",  # Should distinguish 0 from NaN
            "Which column has no missing values?"  # Only 'c'
        ]
        
        for q in questions:
            response = self.chat.chat(q)
            print(f"Q: {q}")
            print(f"A: {response[:150]}...")
            print()
    
    def test_statistical_traps(self):
        """Test statistical edge cases"""
        df = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],  # No variance
            'perfect_correlation_x': [1, 2, 3, 4, 5],
            'perfect_correlation_y': [2, 4, 6, 8, 10],  # y = 2x
            'outlier': [1, 1, 1, 1, 1000],  # Extreme outlier
            'bimodal': [1, 1, 1, 10, 10, 10]  # Bimodal distribution
        })
        self.chat.set_dataframe(df)
        
        print("\nTesting Statistical Traps:")
        print("-" * 50)
        
        traps = [
            "What's the standard deviation of the constant column?",  # Should be 0
            "Is the outlier column normally distributed?",  # Definitely not
            "What's the correlation between perfect_correlation_x and y?",  # Should be 1.0
            "What's the median vs mean of the outlier column?",  # Very different
            "Describe the distribution of bimodal"  # Should mention two peaks
        ]
        
        for q in traps:
            response = self.chat.chat(q)
            print(f"Q: {q}")
            print(f"A: {response[:200]}...")
            print()
    
    def test_context_switching(self):
        """Test if context gets confused when switching datasets"""
        # First dataset
        df1 = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'price': [10, 20, 30]
        })
        self.chat.set_dataframe(df1)
        response1 = self.chat.chat("What products do we have?")
        
        # Second dataset (completely different)
        df2 = pd.DataFrame({
            'employee': ['John', 'Jane', 'Bob'],
            'salary': [50000, 60000, 55000]
        })
        self.chat.set_dataframe(df2)
        response2 = self.chat.chat("What products do we have?")  # Ask about products again
        
        print("\nTesting Context Switching:")
        print("-" * 50)
        print("First dataset had products A, B, C")
        print("Second dataset has employees, no products")
        print(f"Response after switch: {response2[:200]}")
        
        # Check if it incorrectly references the first dataset
        mentions_abc = any(x in response2 for x in ['A', 'B', 'C', 'product'])
        print(f"Incorrectly references old data: {'✗ Yes' if mentions_abc else '✓ No'}")
    
    def test_calculation_accuracy(self):
        """Test precise calculations"""
        df = pd.DataFrame({
            'values': [1.111, 2.222, 3.333, 4.444, 5.555]
        })
        self.chat.set_dataframe(df)
        
        print("\nTesting Calculation Accuracy:")
        print("-" * 50)
        
        # True mean is 3.333
        response = self.chat.chat("What is the EXACT mean of values? Give me all decimal places.")
        print(f"Question: Exact mean (should be 3.333)")
        print(f"Response: {response}")
        
        # Check if it has the right number
        has_correct = "3.333" in response
        print(f"Correct answer: {'✓' if has_correct else '✗'}")
    
    def run_all_tricky_tests(self):
        """Run all tricky scenario tests"""
        print("="*70)
        print("TRICKY REAL-WORLD SCENARIO TESTS")
        print("="*70)
        
        self.test_ambiguous_questions()
        self.test_column_confusion()
        self.test_data_type_assumptions()
        self.test_missing_value_traps()
        self.test_statistical_traps()
        self.test_context_switching()
        self.test_calculation_accuracy()
        
        print("\n" + "="*70)
        print("Review the outputs above for potential issues!")
        print("="*70)


# Additional test queries that often break chatbots
GOTCHA_QUERIES = [
    # Reference confusion
    "Compare this to the previous dataset",  # What previous dataset?
    "Like I said before...",  # No previous context
    "Go back to my first question",  # Which question?
    
    # Calculation traps  
    "Divide sales by zero values",
    "What's the log of negative numbers?",
    "Calculate percentage of a text column",
    
    # Impossible requests
    "Predict next year's sales",  # No model, just data
    "Which employee will quit?",  # Can't predict from data
    "Tell me the customer names",  # When no name column exists
    
    # Format confusion
    "Show me in Excel format",
    "Create a PowerBI dashboard",
    "Write SQL to query this",
    
    # Meta questions
    "Are you sure?",
    "Is that correct?",
    "Can you double-check?",
    "What if you're wrong?",
    
    # Boundary pushing
    "Tell me everything",
    "Analyze all possible combinations",
    "Check every correlation",
    "Find all patterns"
]

if __name__ == "__main__":
    tester = TrickyScenarioTests()
    tester.run_all_tricky_tests()