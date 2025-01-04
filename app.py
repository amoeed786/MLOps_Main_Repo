from transformers import pipeline
import gradio as gr

# Load the sentiment analysis pipeline
model = pipeline("sentiment-analysis")

def predict_sentiment(user_input):
    if not user_input.strip():
        return "Please enter some text to analyze."
    
    result = model(user_input)[0]
    return f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})"
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter text to analyze...", label="User Input"),
    outputs="text",
    title="Sentiment Analysis Tool",
    description="This tool predicts whether the sentiment of the given text is Positive, Negative, or Neutral.",
    examples=[
        "I love spending time with my family!",
        "This weather is terrible.",
        "The movie was okay, not great but not bad."
    ]
)

interface.launch(share=True)
