import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import { GraphView } from "./graph_view";
import { SelectionView } from "./selection_view";
import "./widget.css";

const render = createRender(() => {
  const [alpha, setAlpha] = useModelState<number>("alpha");

  return (
    <div className="topo_widget">
      <div className="widget-header">
        <div className="header-content">
          <h1>TopoBridge</h1>
          <p className="header-description">
            Visualize how nodes position themselves between{" "}
            <span className="highlight attribute">attribute similarity</span>{" "}
            and <span className="highlight topology">network connections</span>
          </p>
        </div>
        <div className="header-icon">
          <svg
            viewBox="0 0 24 24"
            width="36"
            height="36"
            stroke="#4299e1"
            strokeWidth="1.5"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="8" cy="9" r="2" />
            <circle cx="15" cy="6" r="2" />
            <circle cx="18" cy="14" r="2" />
            <circle cx="10" cy="16" r="2" />
            <path d="M8 11 L 10 14" />
            <path d="M10 16 L 15 8" />
            <path d="M16 7 L 18 12" />
          </svg>
        </div>
      </div>

      <div className="embedding-control">
        <div className="embedding-label">
          <span className="embedding-type attribute">Attribute-based</span>
          <span className="embedding-value">{alpha.toFixed(2)}</span>
          <span className="embedding-type topology">Topology-based</span>
        </div>

        <div className="slider-container">
          <input
            id="alpha-slider"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            className="styled-slider"
          />
          <div className="slider-markers">
            <span className="marker">0.0</span>
            <span className="marker">0.5</span>
            <span className="marker">1.0</span>
          </div>
        </div>

        <p className="slider-explanation">
          Move the slider to balance between node attributes (left) and network
          structure (right)
        </p>
      </div>

      <div className="data-container">
        <GraphView />
        <SelectionView />
      </div>
    </div>
  );
});

export default { render };
