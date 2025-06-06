Personality Prediction Pipeline Documentation

⸻

Pipeline Overview

This pipeline preprocesses the Extrovert vs. Introvert Behavior dataset to prepare it for machine learning modeling. It handles categorical encoding, robust scaling, and missing value imputation. Outlier treatment using the Tukey method is available but was not applied in this version.

⸻

Pipeline Diagram

(See screenshot from notebook on Github)

⸻

Step-by-Step Design Choices

1. Stage Fear Mapping (map_stage_fear)
	•	Transformer: CustomMappingTransformer('Stage_fear', {'No': 0, 'Yes': 1})
	•	Design Choice: Convert Yes/No responses to binary 0/1
	•	Rationale: Enables model to interpret categorical feature numerically without adding dimensionality

2. Drained After Socializing Mapping (map_drained)
	•	Transformer: CustomMappingTransformer('Drained_after_socializing', {'No': 0, 'Yes': 1})
	•	Design Choice: Binary encoding of Yes/No values
	•	Rationale: Same as above, converting categorical data into machine-readable format

3. Time Spent Alone Scaling (scale_time_alone)
	•	Transformer: CustomRobustTransformer(target_column='Time_spent_Alone')
	•	Design Choice: Apply robust scaling to reduce influence of outliers
	•	Rationale: Time spent alone may vary widely; robust scaling ensures extreme values don’t skew model

4. Social Event Attendance Scaling (scale_attendance)
	•	Transformer: CustomRobustTransformer(target_column='Social_event_attendance')
	•	Design Choice: Same as above
	•	Rationale: Keeps event attendance values on comparable scale with other features

5. Going Outside Scaling (scale_outside)
	•	Transformer: CustomRobustTransformer(target_column='Going_outside')
	•	Design Choice: Same robust scaling
	•	Rationale: Daily activity levels can have high variance

6. Friends Circle Size Scaling (scale_friends)
	•	Transformer: CustomRobustTransformer(target_column='Friends_circle_size')
	•	Design Choice: Robust transformation to preserve outlier-resilient representation
	•	Rationale: Friends circle sizes vary, but shouldn’t dominate other feature effects

7. Post Frequency Scaling (scale_posting)
	•	Transformer: CustomRobustTransformer(target_column='Post_frequency')
	•	Design Choice: Scaled like all other numeric features
	•	Rationale: Keeps post frequency feature proportional and interpretable

8. Imputation (impute)
	•	Transformer: CustomKNNTransformer(n_neighbors=5)
	•	Design Choice: KNN imputation using 5 nearest neighbors
	•	Rationale: Fills missing values based on feature similarity; balances local detail with generality

⸻

Pipeline Execution Order Rationale
	1.	Categorical encoding first: Makes categorical features usable for downstream transformations
	2.	Outlier removal skipped for now: Preserves data richness for behavior analysis
	3.	Scaling before imputation: Ensures distances in KNN are based on comparable feature ranges
	4.	Imputation last: Fills in any missing values after all transformations

⸻

Performance Considerations
	•	RobustScaler chosen due to potential variance in user behavior (resistant to outliers)
	•	KNN imputation preferred to retain inter-feature relationships over simple imputation
	•	Simple mappings for binary fields reduce dimensionality and improve interpretability