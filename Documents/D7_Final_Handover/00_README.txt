GROUP 74 — COMP3850 PACE
D7 FINAL HANDOVER — DOCUMENTATION
Defensive Privacy-Preserving Record Linkage (PPRL)

This folder holds the final-handover documentation. The prototype code,
datasets and results are in the repository root (see START_HERE.txt).

Contents:
  Cybersecurity_Project_Documentation.docx  - NEW: the 6 required cybersecurity sections
  Security_Test_Plan_and_Reports.docx       - NEW: 15 test cases + testing reports
  Recommendations_and_Monitoring_Plan.docx  - NEW: recommendations + monitoring plan
  Project_Plan_revised.docx                 - risk owners/residual status; corrected handover wording
  Scoping_Document_revised.docx             - corrected cover version/date; evidence-based privacy claims
  Test_Cases.docx                           - test-case catalogue
  Team_Retrospective.docx                   - reflection on implementation, teamwork, testing
  User_Manual.docx                          - user / training manual
  Scripts_and_Execution.docx                - how to run the prototype
  Prototype_Final_Results.docx              - full experimental results
  architecture.png, dataflow.png            - architecture and data-flow / trust-boundary diagrams

To reproduce the results, see START_HERE.txt in the repository root.
Repository: https://github.com/Oslimy/group-74

Reproducibility: In the structured-data experiment (exp_d4.py), the threshold, Logistic Regression, SVM and XGBoost results reproduce exactly; only Random Forest is stochastic and can vary up to ~0.07 F1 between runs - it is the weakest under noise either way. In the text experiment (text_experiment.py), the no-noise row reproduces exactly; the noised rows (eps=7 and eps=3) can vary up to ~0.014 F1 because that perturbation is not seeded.
