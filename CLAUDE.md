# Project constraints
- Deep model framework: Keras/TensorFlow only. Never add PyTorch.
- Function signatures in src/ are team contracts; don't change them without agreeing with the owner.
- main is protected: all changes go through a pull request.
- Commits: no Claude Co-Authored-By trailer.
- Own data is video only (4 people, p1-p4, no phone sensors): a separate camera pipeline (MediaPipe keypoints -> Keras CNN, owner Yashaswini) sits next to the phone pipeline. Videos never go in git; evaluate leave-one-person-out.
