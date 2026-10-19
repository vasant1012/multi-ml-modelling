import Card from "../common/Card";
import ProgressBar from "../common/ProgressBar";
import { percent, titleCase } from "../../utils/formatters";

export default function MetricCard({
    label,
    value,
    confidence,
    icon: Icon,
    tone = "primary",
    detail,
}) {
    return (
        <Card className={`metric-card tone-${tone}`}>
            <div className="metric-top">
                <span className="metric-icon" aria-hidden="true">
                    <Icon />
                </span>
                <div className="metric-heading">
                    <span className="eyebrow">{label}</span>
                </div>
            </div>

            <div className="metric-body">
                <strong className="metric-value">
                    {titleCase(String(value))}
                </strong>
                {detail && <span className="metric-detail">{detail}</span>}
            </div>

            {confidence != null && (
                <div className="metric-footer">
                    <div className="metric-confidence">
                        <span>Confidence</span>
                        <b>{percent(confidence)}</b>
                    </div>
                    <ProgressBar value={confidence} tone={tone} />
                </div>
            )}
        </Card>
    );
}
