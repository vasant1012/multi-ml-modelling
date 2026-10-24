import { CircleDot } from "lucide-react";
import Badge from "../common/Badge";
export default function TopBar({ health }) {
    return (
        <header className="topbar">
            <div>
                <h1>Hotel Review Intelligence</h1>
                <p>Turn guest feedback into operational decisions</p>
            </div>
            <div className="topbar-actions">
                <Badge
                    tone={
                        health === "healthy"
                            ? "positive"
                            : health === "offline"
                              ? "negative"
                              : "neutral"
                    }
                >
                    <CircleDot size={14} /> API {health}
                </Badge>
                <span className="model-pill">Local inference</span>
            </div>
        </header>
    );
}
