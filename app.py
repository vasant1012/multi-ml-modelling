import dash
import dash_bootstrap_components as dbc

from dash import html, dcc, Input, Output, State
import plotly.graph_objs as go
import torch

from transformers import AutoTokenizer
from model import HotelReviewMultiTaskModel
from summarizer import summarize_review
from inference import process_review
import warnings

warnings.filterwarnings("ignore")


def predict_review(text):
    local_model_path = "hotel_review_model.pt"
    # # Load tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    # model = RobertaForSequenceClassification.from_pretrained(local_model_path)
    model = HotelReviewMultiTaskModel()
    model.load_state_dict(torch.load(local_model_path))
    model.eval()

    results = process_review(text, model, summarize_review, tokenizer, device)
    return results


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def render_aspect_chart(aspects):
    aspect_names = list(aspects.keys())
    scores = [aspects[a]["score"] for a in aspect_names]
    sentiments = [aspects[a]["sentiment"] for a in aspect_names]

    bar_colors = [
        "gray" if s == "neutral" else "green" if s == "positive" else "red"
        for s in sentiments
    ]

    return dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(
                    x=aspect_names,
                    y=scores,
                    marker_color=bar_colors,
                    text=sentiments,
                    textposition="auto",
                )
            ],
            layout=go.Layout(title="Aspect Scores & Sentiments", height=400),
        ),
        id="aspect-graph",
    )


app.layout = html.Div([
    dbc.Container(
        [
            html.H2("Hotel Review Analyzer", className="text-center my-3"),
            html.Br(),
            dbc.Textarea(
                id="text-input",
                placeholder="Enter your hotel review",  # NOQA E501
                value="".format(),
                style={"width": "100%", "height": 80},
            ),  # NOQA E501
            html.Br(),
            dbc.Button(
                "Analyze", id="submit-btn", n_clicks=0, style={"height": "40px"}
            ),
            html.Br(),
            html.Hr(),
        ],
        fluid=True,
    ),
    html.Div(
        id="output-area",
        children=[
            html.H4("Review Summary"),
            html.Br(),
            dbc.Alert("", id="summary-alert", color="light"),
            html.Hr(),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Rating"),
                            html.Div(id="rating-section"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.H5("Sentiment"),
                            dbc.Badge(
                                id="sentiment-badge",
                                color="secondary",
                                className="mb-3",
                                pill=True,
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.H5("Emotions"),
                            html.Div(id="emotions-badges", className="mb-3"),
                        ],
                        width=4,
                    ),
                ],
                className="mb-4",
            ),
            html.Hr(),
            html.Br(),
            html.H5("Aspect Analysis"),
            html.Div(id="aspect-chart"),
        ],
    ),
])


@app.callback(
    Output("summary-alert", "children"),
    Output("rating-section", "children"),
    Output("sentiment-badge", "children"),
    Output("sentiment-badge", "color"),
    Output("emotions-badges", "children"),
    Output("aspect-chart", "children"),
    Input("submit-btn", "n_clicks"),
    State("text-input", "value"),
    prevent_initial_call=False,
)
def update_output(n_clicks, input_text):
    if not input_text and n_clicks > 0:
        return dash.no_update

    result = predict_review(input_text)

    # Rating gauge
    rating_stars = html.Span(
        "★" * int(result["rating"])
        + "☆" * (5 - int(result["rating"])),  # Example: 4 filled stars, 1 empty
        style={"color": "gold", "height": "30px"},
    )

    # Emotions badges
    emotion_badges = [
        dbc.Badge(e, color="info", className="me-1") for e in result["emotions"]
    ]

    # Sentiment badge color
    sentiment_color = {
        "positive": "success",
        "negative": "danger",
        "neutral": "secondary",
    }.get(result["sentiment"], "light")

    # Aspect chart
    aspect_chart = render_aspect_chart(result["aspects"])

    return (
        result["summary"],
        rating_stars,
        result["sentiment"].capitalize(),
        sentiment_color,
        emotion_badges,
        aspect_chart,
    )


if __name__ == "__main__":
    app.run(debug=False)
