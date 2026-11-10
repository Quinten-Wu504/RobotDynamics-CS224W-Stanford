# Stage 0: setup notes

## Repository

Public repository: [Quinten-Wu504/RobotDynamics-CS224W-Stanford](https://github.com/Quinten-Wu504/RobotDynamics-CS224W-Stanford). Default branch: `main`.

The name follows the topic-and-school pattern used in these earlier course repositories:

- `ClassicalVision-Coursework-Tongji`
- `ImageClassification-Coursework-Tongji`
- `ImageProcessing-Coursework-Tongji`

All three are personal repositories, not forks. This project adds `CS224W` to identify the course.

## Files

The project lives in its own directory. Course PDFs and the local execution plan stay in the parent directory and are not tracked.

- `README.md`: project scope and setup status.
- `.gitignore`: local environments, credentials, datasets, checkpoints, and course files.
- `docs/progress.md`: roadmap and current work.
- `docs/stage_0.md`: these setup notes.

## Checks

- Reviewed the initial four-file diff; whitespace checks passed.
- Checked ignore rules against sample credential, environment, data, checkpoint, and course-file paths.
- Pushed the initial commit, `7bdc4d0`, and confirmed that local and remote `main` matched.

## Still to do

No environment or experiment has been run yet. The next stage covers hardware and dependency checks, the task protocol, and a minimal environment. Data counts, splits, action semantics, and prediction timing will need verification before training.
