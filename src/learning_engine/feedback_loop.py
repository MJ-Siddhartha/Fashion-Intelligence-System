from typing import List, Callable, Any

class FeedbackLoop:
    """
    Feedback loop implementation to collect, manage, and process feedback.
    """

    def __init__(self, feedback_processor: Callable[[str], Any] = None):
        """
        Initialize the feedback loop.

        Args:
            feedback_processor (Callable, optional): A callback function to process individual feedback.
                                                     Defaults to None.
        """
        self.feedback_queue: List[str] = []
        self.feedback_processor = feedback_processor

    def add_feedback(self, feedback: str):
        """
        Add feedback to the queue.

        Args:
            feedback (str): Feedback text to be added.
        """
        if not feedback or not isinstance(feedback, str):
            print("Invalid feedback: Feedback must be a non-empty string.")
            return

        self.feedback_queue.append(feedback)
        print(f"Feedback added: {feedback}")

    def process_feedback(self):
        """
        Process all feedback in the queue using the provided feedback processor.
        """
        if not self.feedback_queue:
            print("No feedback to process.")
            return

        while self.feedback_queue:
            feedback = self.feedback_queue.pop(0)
            print(f"Processing feedback: {feedback}")

            # If a feedback processor is provided, use it
            if self.feedback_processor:
                try:
                    result = self.feedback_processor(feedback)
                    print(f"Processed feedback result: {result}")
                except Exception as e:
                    print(f"Error processing feedback: {e}")
            else:
                print("No feedback processor provided. Feedback skipped.")

    def view_pending_feedback(self) -> List[str]:
        """
        View all feedback currently in the queue.

        Returns:
            List[str]: List of pending feedback.
        """
        return self.feedback_queue

    def clear_feedback_queue(self):
        """
        Clear all pending feedback in the queue.
        """
        self.feedback_queue.clear()
        print("Feedback queue cleared.")


# Example Integration with a Feedback Processor
if __name__ == "__main__":
    def sample_feedback_processor(feedback: str) -> dict:
        """
        Example feedback processor function to classify feedback.

        Args:
            feedback (str): Feedback text.

        Returns:
            dict: Feedback classification result.
        """
        positive_keywords = ["good", "great", "excellent", "positive"]
        negative_keywords = ["bad", "poor", "negative"]

        feedback_lower = feedback.lower()
        if any(keyword in feedback_lower for keyword in positive_keywords):
            classification = "positive"
        elif any(keyword in feedback_lower for keyword in negative_keywords):
            classification = "negative"
        else:
            classification = "neutral"

        return {"feedback": feedback, "classification": classification}

    # Instantiate FeedbackLoop with the sample feedback processor
    feedback_loop = FeedbackLoop(feedback_processor=sample_feedback_processor)

    # Add feedback
    feedback_loop.add_feedback("The system works great!")
    feedback_loop.add_feedback("The response time is too slow.")
    feedback_loop.add_feedback("I'm not sure about this feature.")

    # View pending feedback
    print("Pending Feedback:", feedback_loop.view_pending_feedback())

    # Process feedback
    feedback_loop.process_feedback()

    # Clear feedback queue
    feedback_loop.clear_feedback_queue()
