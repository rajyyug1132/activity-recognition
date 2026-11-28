# Project constraints
- Deep model framework: Keras/TensorFlow only. Never add PyTorch.
- Function signatures in src/ are team contracts; don't change them without agreeing with the owner.
- main is protected: all changes go through a pull request.
- Commits: no Claude Co-Authored-By trailer.
- Own data is video only (5 people, p1-p5, no phone sensors; p5 has one walking clip so is training-only): a separate camera pipeline (MediaPipe keypoints -> Keras CNN, owner Yashaswini) sits next to the phone pipeline. Videos never go in git; evaluate leave-one-person-out.
- UI: follow DESIGN.md (v3). Colour means an activity and nothing else. No tracked-uppercase eyebrows, side-stripe borders or gradient text; hex values only in app/styles/theme.css and .streamlit/config.toml.
