# Metrics (evaluators)

Each file defines an evaluator for a task with a specific setting. Each evaluator takes **predictions** (a list of predictions) and **golds** (a list of the gold data item, and each gold data item is a dict) and returns a dict as the **evaluation result**.

You can add new evaluators here for the evaluation of a new task or an existing task with a new setting. If you use the `../third_party/` directory for the evaluator, please **add them into** the `../third_party` directory, and specify their link in `.gitsubmodule`, which enables recursive cloning. 
