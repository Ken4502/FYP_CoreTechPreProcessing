import cv2
from core.workflows import WorkflowRunner

def test():
    runner = WorkflowRunner()
    img = cv2.imread('samples/diseased_leaf_sample.png')
    assert img is not None, "Failed to load sample image"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Running all 4 workflows...")
    res = runner.run_workflow_adaptive(img_rgb)
    print("Winner:", res["winner_name"])
    print(f"Winner Confidence: {res['winner_confidence']}%")
    for k, v in res["workflows"].items():
        print(f" - {k}: {v['eval']['confidence_pct']}% | Latency: {v['latency_ms']} ms")
    print("Pipeline integration test completed successfully!")

if __name__ == "__main__":
    test()
