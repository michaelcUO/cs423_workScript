Personality Prediction Pipeline Documentation

⸻

Pipeline Overview

This pipeline preprocesses the Extrovert vs. Introvert Behavior dataset to prepare it for machine learning modeling. It handles categorical encoding, outlier removal via the Tukey method, robust feature scaling, and missing value imputation.

⸻

Pipeline Diagram

(See screenshot from notebook on GitHub)

⸻

Step-by-Step Design Choices

1. Stage Fear Mapping (map_stage_fear)
	•	Transformer: CustomMappingTransformer('Stage_fear', {'No': 0, 'Yes': 1})
	•	Rationale: Converts categorical Yes/No input into binary numeric format.

2. Drained After Socializing Mapping (map_drained)
	•	Transformer: CustomMappingTransformer('Drained_after_socializing', {'No': 0, 'Yes': 1})
	•	Rationale: Binary encoding to ensure compatibility with numeric modeling steps.

⸻

Outlier Treatment with Tukey Transformers

3. Time Spent Alone (tukey_time_alone)
	•	Transformer: CustomTukeyTransformer(target_column='Time_spent_Alone', fence='outer')
	•	Rationale: Removes extreme behavioral outliers while retaining normal variation.

4. Social Event Attendance (tukey_attendance)
	•	Transformer: CustomTukeyTransformer(target_column='Social_event_attendance', fence='outer')
	•	Rationale: Smooths out rare cases of very low or very high attendance.

5. Going Outside (tukey_outside)
	•	Transformer: CustomTukeyTransformer(target_column='Going_outside', fence='outer')
	•	Rationale: Reduces influence of extreme introvert/extrovert daily behavior.

6. Friends Circle Size (tukey_friends)
	•	Transformer: CustomTukeyTransformer(target_column='Friends_circle_size', fence='outer')
	•	Rationale: Caps outliers without distorting the distribution of social reach.

7. Post Frequency (tukey_posting)
	•	Transformer: CustomTukeyTransformer(target_column='Post_frequency', fence='outer')
	•	Rationale: Reduces impact of edge cases (e.g., people who post excessively or not at all).

⸻

Robust Scaling

8. Time Spent Alone (scale_time_alone)
	•	Transformer: CustomRobustTransformer(target_column='Time_spent_Alone')
	•	Rationale: Handles remaining skew while preserving behavioral signal.

9. Social Event Attendance (scale_attendance)
	•	Transformer: CustomRobustTransformer(target_column='Social_event_attendance')

10. Going Outside (scale_outside)
	•	Transformer: CustomRobustTransformer(target_column='Going_outside')

11. Friends Circle Size (scale_friends)
	•	Transformer: CustomRobustTransformer(target_column='Friends_circle_size')

12. Post Frequency (scale_posting)
	•	Transformer: CustomRobustTransformer(target_column='Post_frequency')

⸻

Imputation

13. KNN Imputer (impute)
	•	Transformer: CustomKNNTransformer(n_neighbors=5)
	•	Rationale: Estimates missing values using nearby data points for higher fidelity than mean/median.

⸻

Pipeline Execution Order Rationale
	1.	Categorical mapping prepares features for numeric transformations.
	2.	Tukey transformers handle outliers before scaling.
	3.	Robust scaling ensures consistent feature ranges.
	4.	KNN imputation fills in missing values based on transformed feature space.

⸻

Performance Considerations
	•	Tukey fences improve model robustness by reducing noise from extremes.
	•	RobustScaler complements Tukey by further minimizing outlier influence.
	•	KNN imputation retains structure in user behavior patterns for better generalization.

⸻