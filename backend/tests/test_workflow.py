import sys

sys.path.append("C:\\Users\\vasant.rajadhvax\\LocalBin_vasant.rajadhyax\\Projects\\multi-ml-modelling")

from backend.graph import review_graph
from backend.logger import logger


def test_graph():

    result = review_graph.invoke(
        {
            "review_text":
                "Amazing hotel and great staff"
        }
    )

    logger.info("Graph result: %s", result)

if __name__ == "__main__":
    test_graph()