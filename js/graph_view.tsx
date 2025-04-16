import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { useModelState } from "@anywidget/react";

export function GraphView() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNodes, setSelectedNodes] =
    useModelState<string[]>("selected_nodes");

  const [nodePositions, setNodePositions] =
    useModelState<Record<string, { x: number; y: number }>>("node_positions");

  const [nodeTravelDistances] = useModelState<Record<string, number>>(
    "node_travel_distances"
  );

  // Add a new state for the graph edges
  const [graphEdges] = useModelState<Array<[string, string]>>("graph_edges");

  // Create scatterplot when nodePositions changes
  useEffect(() => {
    if (!svgRef.current || !nodePositions) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous content

    // Get the actual dimensions of the SVG element
    const svgNode = svg.node();
    if (!svgNode) return;

    const boundingRect = svgNode.getBoundingClientRect();
    const width = boundingRect.width;
    const height = boundingRect.height;
    const margin = 40;

    // Convert object to array of nodes
    const nodes = Object.entries(nodePositions).map(([id, pos]) => ({
      id,
      x: pos.x,
      y: pos.y,
      selected: selectedNodes.includes(id),
      travelDistance: nodeTravelDistances?.[id] || 0,
    }));

    // Create scales
    const xExtent = d3.extent(nodes, (d) => d.x) as [number, number];
    const yExtent = d3.extent(nodes, (d) => d.y) as [number, number];

    const xScale = d3
      .scaleLinear()
      .domain([xExtent[0] - 0.1, xExtent[1] + 0.1])
      .range([margin, width - margin]);

    const yScale = d3
      .scaleLinear()
      .domain([yExtent[0] - 0.1, yExtent[1] + 0.1])
      .range([height - margin, margin]);

    // Create a color scale for the travel distance
    const travelDistances = nodes.map((n) => n.travelDistance);
    const minDistance = d3.min(travelDistances) || 0;
    const maxDistance = d3.max(travelDistances) || 1;

    const colorScale = d3
      .scaleSequential()
      .domain([minDistance, maxDistance])
      .interpolator(d3.interpolateBlues);

    // Create a container for the edges and draw them BEFORE the nodes
    if (graphEdges && graphEdges.length > 0) {
      const edgesGroup = svg.append("g").attr("class", "edges");

      // Create a nodePositionsMap for quick lookups
      const nodePositionsMap = new Map(
        Object.entries(nodePositions).map(([id, pos]) => [id, pos])
      );

      // Draw edges
      edgesGroup
        .selectAll("line")
        .data(graphEdges)
        .enter()
        .append("line")
        .attr("x1", (d) => xScale(nodePositionsMap.get(d[0])?.x || 0))
        .attr("y1", (d) => yScale(nodePositionsMap.get(d[0])?.y || 0))
        .attr("x2", (d) => xScale(nodePositionsMap.get(d[1])?.x || 0))
        .attr("y2", (d) => yScale(nodePositionsMap.get(d[1])?.y || 0))
        .attr("stroke", "#e2e8f0") // Light gray color
        .attr("stroke-width", 0.5)
        .attr("opacity", 0.6)
        .attr("stroke-linecap", "round");
    }

    // Create a container for the points
    const pointsGroup = svg.append("g").attr("class", "points");

    // Sort nodes to draw unselected nodes first, selected nodes last
    const sortedNodes = [...nodes].sort((a, b) => {
      return a.selected === b.selected ? 0 : a.selected ? 1 : -1;
    });

    // Draw points with the sorted data
    pointsGroup
      .selectAll("circle")
      .data(sortedNodes)
      .enter()
      .append("circle")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", (d) => (d.selected ? 7 : 5))
      .attr("fill", (d) =>
        d.selected ? "#f56565" : colorScale(d.travelDistance)
      )
      .attr("stroke", (d) => (d.selected ? "#e53e3e" : "#2b6cb0"))
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("click", (event, d) => {
        if (!isDragging) {
          if (selectedNodes.includes(d.id)) {
            setSelectedNodes(selectedNodes.filter((id) => id !== d.id));
          } else {
            setSelectedNodes([...selectedNodes, d.id]);
          }
        }
      });

    // Create a tooltip div that stays hidden until hover
    const tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "node-tooltip")
      .style("position", "absolute")
      .style("visibility", "hidden")
      .style("background-color", "white")
      .style("border", "1px solid #ccc")
      .style("border-radius", "4px")
      .style("padding", "8px")
      .style("font-size", "12px")
      .style("box-shadow", "0 2px 4px rgba(0,0,0,0.1)")
      .style("pointer-events", "none")
      .style("z-index", "100");

    // Add mouse events for the tooltips to the circle elements
    pointsGroup
      .selectAll("circle")
      .on("mouseover", (event, d) => {
        // Format travel distance for display
        const travelText = d.travelDistance.toFixed(3);

        // Build tooltip content
        let content = `<strong>Node ID:</strong> ${d.id}<br/>`;
        content += `<strong>Travel Distance:</strong> ${travelText}`;

        // Show and position the tooltip
        tooltip
          .style("visibility", "visible")
          .html(content)
          .style("left", `${event.pageX + 10}px`)
          .style("top", `${event.pageY - 10}px`);
      })
      .on("mousemove", (event) => {
        // Move tooltip with the mouse
        tooltip
          .style("left", `${event.pageX + 10}px`)
          .style("top", `${event.pageY - 10}px`);
      })
      .on("mouseout", () => {
        // Hide tooltip when mouse leaves the node
        tooltip.style("visibility", "hidden");
      });

    // Add a color legend
    const legendWidth = Math.min(200, width * 0.4); // Constrain legend width for small screens
    const legendHeight = 20;
    const legendX = margin;
    const legendY = height - margin - legendHeight - 20; // Position above the bottom

    const defs = svg.append("defs");
    const gradient = defs
      .append("linearGradient")
      .attr("id", "travel-distance-gradient")
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "0%");

    gradient
      .append("stop")
      .attr("offset", "0%")
      .attr("stop-color", colorScale(minDistance));

    gradient
      .append("stop")
      .attr("offset", "100%")
      .attr("stop-color", colorScale(maxDistance));

    // Add a background for the legend to make it stand out
    svg
      .append("rect")
      .attr("x", legendX - 5) // Add padding around the legend
      .attr("y", legendY - 20) // Cover title and legend
      .attr("width", legendWidth + 10)
      .attr("height", legendHeight + 35) // Make sure it covers the text below
      .attr("rx", 4) // Rounded corners
      .attr("ry", 4)
      .attr("fill", "rgba(255, 255, 255, 0.9)") // Slightly transparent white
      .attr("stroke", "#e2e8f0")
      .attr("stroke-width", 1);

    svg
      .append("rect")
      .attr("x", legendX)
      .attr("y", legendY)
      .attr("width", legendWidth)
      .attr("height", legendHeight)
      .style("fill", "url(#travel-distance-gradient)");

    svg
      .append("text")
      .attr("x", legendX)
      .attr("y", legendY - 5)
      .style("font-size", "10px")
      .style("font-weight", "bold")
      .text("Travel Distance (α=0 to α=1)");

    svg
      .append("text")
      .attr("x", legendX)
      .attr("y", legendY + legendHeight + 12)
      .style("font-size", "9px")
      .style("text-anchor", "start")
      .text(`Min: ${minDistance.toFixed(2)}`);

    svg
      .append("text")
      .attr("x", legendX + legendWidth)
      .attr("y", legendY + legendHeight + 12)
      .style("font-size", "9px")
      .style("text-anchor", "end")
      .text(`Max: ${maxDistance.toFixed(2)}`);

    // Create a selection rectangle for bounding box selection
    const selectionRect = svg
      .append("rect")
      .attr("class", "selection-rect")
      .style("fill", "rgba(66, 153, 225, 0.2)")
      .style("stroke", "#4299e1")
      .style("stroke-width", 1)
      .style("stroke-dasharray", "4,4")
      .style("display", "none");

    let startX: number;
    let startY: number;
    let isDragging = false;

    svg.on("mousedown", (event) => {
      if (event.target.tagName !== "circle") {
        const [x, y] = d3.pointer(event);
        startX = x;
        startY = y;
        isDragging = true;

        selectionRect
          .attr("x", startX)
          .attr("y", startY)
          .attr("width", 0)
          .attr("height", 0)
          .style("display", null);
      }
    });

    svg.on("mousemove", (event) => {
      if (!isDragging) return;

      const [x, y] = d3.pointer(event);

      const width = Math.abs(x - startX);
      const height = Math.abs(y - startY);
      const boxX = x < startX ? x : startX;
      const boxY = y < startY ? y : startY;

      selectionRect
        .attr("x", boxX)
        .attr("y", boxY)
        .attr("width", width)
        .attr("height", height);
    });

    svg.on("mouseup", (event) => {
      if (!isDragging) return;

      const selBox = {
        x1: parseFloat(selectionRect.attr("x")),
        y1: parseFloat(selectionRect.attr("y")),
        x2:
          parseFloat(selectionRect.attr("x")) +
          parseFloat(selectionRect.attr("width")),
        y2:
          parseFloat(selectionRect.attr("y")) +
          parseFloat(selectionRect.attr("height")),
      };

      const selectedIds = nodes
        .filter((node) => {
          const nodeX = xScale(node.x);
          const nodeY = yScale(node.y);
          return (
            nodeX >= selBox.x1 &&
            nodeX <= selBox.x2 &&
            nodeY >= selBox.y1 &&
            nodeY <= selBox.y2
          );
        })
        .map((node) => node.id);

      if (selectedIds.length > 0) {
        setSelectedNodes(selectedIds);
      } else {
        const dragDistance = Math.sqrt(
          Math.pow(selBox.x2 - selBox.x1, 2) +
            Math.pow(selBox.y2 - selBox.y1, 2)
        );
        if (dragDistance < 5) {
          setSelectedNodes([]);
        }
      }

      selectionRect.style("display", "none");
      isDragging = false;
    });

    svg.on("mouseleave", () => {
      if (isDragging) {
        selectionRect.style("display", "none");
        isDragging = false;
      }
    });

    // Make sure to clean up tooltip when component unmounts
    return () => {
      d3.select("body").selectAll(".node-tooltip").remove();
    };
  }, [
    nodePositions,
    selectedNodes,
    setSelectedNodes,
    nodeTravelDistances,
    graphEdges,
  ]);

  return (
    <div className="graph-view">
      <svg ref={svgRef} width="100%" height="100%"></svg>
    </div>
  );
}
