from Topo_Algo import Topo_Algo
import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils import resample
import statsmodels.api as sm
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC


class LP_Supervised_Algo(Topo_Algo):
    def __init__(self, case_dict, graph_object, excel_path, attack_path, round_case=None):
        super().__init__(case_dict, graph_object, excel_path, attack_path, round_case)
        
    
    def execute_attack(self):
        model_name = self.case_dict["Model"]
        algo_name = self.case_dict["Algo"]  
        
        model_name = f"univariate_{model_name}"   
        
        if hasattr(self, model_name) and callable(getattr(self, model_name)):
            # Get the method
            method = getattr(self, model_name)
            # Call the method with the algo_type argument
            method(algo=algo_name)
        else:
            raise ValueError(f"No method found for {model_name}_{algo_name}.")
            
    
    def _data_prepare(self, data):
        num_nodes = data.shape[0]
        
        # Check if the data matrix is symmetric
        is_symmetric = np.allclose(data, data.T)
        
        # Prepare features and labels
        X = []
        y = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Skip diagonal (self-connection)
                    # If the matrix is symmetric, we only process one side of the diagonal
                    if is_symmetric and i < j:
                        # Feature vector for node pair (i, j)
                        feature_vector = [data[i, j]]
                        X.append(feature_vector)
                        
                        # Label: 1 if there is an edge between nodes i and j, 0 otherwise
                        if self.graph.has_edge(i, j):
                            y.append(1)  # Direct link in the graph
                        else:
                            y.append(0)  # No direct link in the graph
        
                    # If the matrix is asymmetric, we process both sides
                    elif not is_symmetric:
                        # Feature vector for node pair (i, j)
                        feature_vector = [data[i, j]]
                        X.append(feature_vector)
        
                        # Label: 1 if there is an edge between nodes i and j, 0 otherwise
                        if self.graph.has_edge(i, j):
                            y.append(1)  # Direct link in the graph
                        else:
                            y.append(0)  # No direct link in the graph
        
        # Convert X and y to NumPy arrays
        X = np.array(X)
        y = np.array(y)
        
        return X , y
        
        
    def _classification_report_extract(self, report, test_size, X_train, X_test, model_name, algo_name):
        # Extract metrics from the classification report
        accuracy = report['accuracy']
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']
        weighted_precision = report['weighted avg']['precision']
        weighted_recall = report['weighted avg']['recall']
        weighted_f1 = report['weighted avg']['f1-score']
        
        # Record the results for this split ratio
        self.append_experiment_results(
            self.attack_path, 
            [
                self.case_dict["File_name"], 
                self.case_dict["Metric"], 
                model_name, 
                algo_name, 
                test_size, 
                len(X_train), 
                len(X_test), 
                accuracy, 
                macro_precision, 
                macro_recall, 
                macro_f1, 
                weighted_precision, 
                weighted_recall, 
                weighted_f1
            ]
        )
        
        
    def univariate_logistic(self, algo="balanced"):
        """
        Performs the supervised univariate logistic regression with different algo to deal with the imbalanced label problem: 
        'balanced' (class_weight), 'SMOTE' (oversampling), or 'undersampling'.
        
        Parameters:
            algo (str): The balancing algo to apply. Options are:
                          - "balanced": Uses class_weight='balanced' in logistic regression.
                          - "SMOTE": Applies SMOTE oversampling to the minority class.
                          - "undersampling": Uses random undersampling to balance the classes.
        """
        
        df = self.extract_data_from_directory()
        data = np.array(df.iloc[0:, 0:].values, dtype=float)
        
        X, y = self._data_prepare(data)
        
        test_ratios = [0.2, 0.3]
        
        for test_size in test_ratios:
            # Split the original imbalanced dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            
            if algo == "balanced":
                # Logistic regression with class_weight='balanced'
                model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif algo == "SMOTE":
                # Apply SMOTE to the training data to oversample the minority class
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                model = LogisticRegression()
                model.fit(X_train_resampled, y_train_resampled)
                y_pred = model.predict(X_test)
                
            elif algo == "undersampling":
                # Random undersampling of the majority class in the training data
                majority_class = max(set(y_train), key=list(y_train).count)
                majority_indices = np.where(y_train == majority_class)[0]
                majority_samples = X_train[majority_indices]
                majority_labels = y_train[majority_indices]
                
                undersampler = RandomUnderSampler(random_state=42)
                X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train) #in this case, the number of 1 and 0 label held as 50:50
                model = LogisticRegression()
                model.fit(X_train_resampled, y_train_resampled)
                
                # Combine the test set with the held-out majority samples for evaluation
                X_eval = np.vstack((X_test, majority_samples))
                y_eval = np.hstack((y_test, majority_labels))
                y_pred = model.predict(X_eval)
                
            else:
                raise ValueError("Unsupported method. Choose from 'balanced', 'SMOTE', or 'undersampling'.")
        
            if algo == "undersampling":
                X_eval, y_eval = X_eval, y_eval  # because the number of undersampling test data is very limited, so we include the held-out samples(1 label) 
                                                 # to increase roubustness.
            else:
                X_eval, y_eval = X_test, y_test  # Standard test set evaluation
            
            report = classification_report(y_eval, y_pred, output_dict=True)
            
            self._classification_report_extract(report, test_size, X_train_resampled if algo in ["SMOTE", "undersampling"] else X_train, X_eval, "Logit", algo)
                                  

    def univariate_random_forest(self, algo="balanced"):
        """
        Performs random forest classification with different balancing techniques: 
        'balanced' (class_weight), 'SMOTE' (oversampling), or 'undersampling'.
        """
        
        df = self.extract_data_from_directory()
        data = np.array(df.iloc[0:, 0:].values, dtype=float)
        
        X, y = self._data_prepare(data)
        
        test_ratios = [0.2, 0.3]
        
        for test_size in test_ratios:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            
            if algo == "balanced":
                model = RandomForestClassifier(class_weight='balanced', random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                X_eval, y_eval = X_test, y_test
            elif algo == "SMOTE":
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train_resampled, y_train_resampled)
                y_pred = model.predict(X_test)
                X_eval, y_eval = X_test, y_test
            elif algo == "undersampling":
                undersampler = RandomUnderSampler(random_state=42)
                X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train_resampled, y_train_resampled)
                X_eval = np.vstack((X_test, X_train[y_train == max(set(y_train), key=list(y_train).count)]))
                y_eval = np.hstack((y_test, y_train[y_train == max(set(y_train), key=list(y_train).count)]))
                y_pred = model.predict(X_eval)
            else:
                raise ValueError("Unsupported algo. Choose from 'balanced', 'SMOTE', or 'undersampling'.")
            
            report = classification_report(y_eval, y_pred, output_dict=True)
            self._classification_report_extract(report, test_size, X_train_resampled if algo != "balanced" else X_train, X_eval, "RandomForest", algo)
            
    
    def univariate_svm(self, algo="balanced"):
        """
        Performs SVM classification with different balancing techniques: 
        'balanced' (class_weight), 'SMOTE' (oversampling), or 'undersampling'.
        
        Parameters:
            algo (str): The balancing algorithm to apply. Options are:
                        - "balanced": Uses class_weight='balanced' in SVM.
                        - "SMOTE": Applies SMOTE oversampling to the minority class.
                        - "undersampling": Uses random undersampling to balance the classes.
        """
        
        df = self.extract_data_from_directory()
        data = np.array(df.iloc[0:, 0:].values, dtype=float)
        
        X, y = self._data_prepare(data)
        
        test_ratios = [0.2, 0.3]
        
        for test_size in test_ratios:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            
            if algo == "balanced":
                model = SVC(class_weight='balanced', random_state=42)  # Applies class weight balancing
            else:
                model = SVC(random_state=42)
            
            if algo == "SMOTE":
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                model.fit(X_train_resampled, y_train_resampled)
                y_pred = model.predict(X_test)
                X_eval, y_eval = X_test, y_test
            elif algo == "undersampling":
                undersampler = RandomUnderSampler(random_state=42)
                X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
                model.fit(X_train_resampled, y_train_resampled)
                X_eval = np.vstack((X_test, X_train[y_train == max(set(y_train), key=list(y_train).count)]))
                y_eval = np.hstack((y_test, y_train[y_train == max(set(y_train), key=list(y_train).count)]))
                y_pred = model.predict(X_eval)
            elif algo == "balanced":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                X_eval, y_eval = X_test, y_test
            else:
                raise ValueError("Unsupported algo. Choose from 'balanced', 'SMOTE', or 'undersampling'.")
            
            report = classification_report(y_eval, y_pred, output_dict=True)
            self._classification_report_extract(report, test_size, X_train_resampled if algo != "balanced" else X_train, X_eval, "SVM", algo)
        
        
        
        
         
                                    
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        