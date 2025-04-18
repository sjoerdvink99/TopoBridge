import React, { useEffect, useRef } from "react";
import { useModelState } from "@anywidget/react";
import * as d3 from "d3";

type NodeAttributes = Record<string, Record<string, any>>;

export function SelectionView() {
  const [selectedNodeAttributes] = useModelState<NodeAttributes>(
    "selected_node_attributes"
  );
  const [allNodeAttributes] = useModelState<NodeAttributes>(
    "all_node_attributes"
  );
  const [selectedNodes, setSelectedNodes] =
    useModelState<string[]>("selected_nodes");

  // Fix type errors in handleBackgroundClick function
  const handleBackgroundClick = (e: React.MouseEvent<HTMLDivElement>) => {
    // Only clear if clicking directly on the background (not on a chart)
    if (
      e.target instanceof HTMLDivElement &&
      (e.target as HTMLDivElement).className === "selection-view"
    ) {
      setSelectedNodes([]);
    }
  };

  // Always show the view, even if no nodes are selected
  if (!allNodeAttributes || Object.keys(allNodeAttributes).length === 0) {
    return <div className="selection-view empty">No graph data available</div>;
  }

  // Group all attribute values by their key for aggregation
  const attributeGroups: Record<string, any[]> = {};
  const attributeTypes: Record<
    string,
    "numeric" | "categorical" | "binary" | "high_cardinality"
  > = {};
  const allAttributeGroups: Record<string, any[]> = {};
  const selectedAttributeGroups: Record<string, any[]> = {};

  // Extract all values for each attribute (from all nodes)
  if (allNodeAttributes) {
    Object.values(allNodeAttributes).forEach((attrs) => {
      Object.entries(attrs).forEach(([key, value]) => {
        if (!allAttributeGroups[key]) {
          allAttributeGroups[key] = [];
        }

        // Try to convert to number
        const numValue = Number(value);
        if (!isNaN(numValue)) {
          allAttributeGroups[key].push(numValue);
        } else {
          allAttributeGroups[key].push(value);
        }
      });
    });
  }

  // Extract values for each attribute (from selected nodes if any)
  if (
    selectedNodeAttributes &&
    Object.keys(selectedNodeAttributes).length > 0
  ) {
    Object.values(selectedNodeAttributes).forEach((attrs) => {
      Object.entries(attrs).forEach(([key, value]) => {
        if (!selectedAttributeGroups[key]) {
          selectedAttributeGroups[key] = [];
        }

        // Try to convert to number
        const numValue = Number(value);
        if (!isNaN(numValue)) {
          selectedAttributeGroups[key].push(numValue);
        } else {
          selectedAttributeGroups[key].push(value);
        }
      });
    });
  }

  // Determine attribute types from all nodes
  Object.entries(allAttributeGroups).forEach(([key, values]) => {
    // Try to infer numeric vs categorical
    const numericValues = values.filter(
      (v) => typeof v === "number" || !isNaN(Number(v))
    );

    if (numericValues.length === values.length) {
      // All values are numeric
      if (
        key === "gender" ||
        key === "public" ||
        isBinaryAttribute(values.map((v) => Number(v)))
      ) {
        attributeTypes[key] = "binary";
      } else {
        attributeTypes[key] = "numeric";
      }
    } else {
      // For categorical data, check cardinality
      const uniqueValues = new Set(values);
      if (uniqueValues.size > 10) {
        attributeTypes[key] = "high_cardinality";
      } else {
        attributeTypes[key] = "categorical";
      }
    }

    // Initialize attributeGroups with all nodes - used for display
    attributeGroups[key] = values;
  });

  // Helper function to check if an attribute is binary (only 0s and 1s)
  function isBinaryAttribute(values: any[]): boolean {
    if (!values || values.length === 0) return false;

    const uniqueValues = new Set(values.map((v) => Number(v)));
    return (
      uniqueValues.size <= 2 &&
      !uniqueValues.has(NaN) &&
      Array.from(uniqueValues).every((v) => v === 0 || v === 1)
    );
  }

  // Create distribution and bar chart components
  const attributeSummaries = Object.entries(attributeGroups).map(
    ([attrName, values]) => {
      const isNumeric = attributeTypes[attrName] === "numeric";
      const isBinary = attributeTypes[attrName] === "binary";
      const isHighCardinality = attributeTypes[attrName] === "high_cardinality";
      const selectedValues = selectedAttributeGroups[attrName] || [];
      const hasSelection = selectedValues.length > 0;

      return (
        <div key={attrName} className="attribute-summary">
          {isNumeric ? (
            <NumericDistribution
              name={attrName}
              values={selectedValues as number[]}
              allValues={values as number[]}
              showSelectionOnly={false}
            />
          ) : isBinary ? (
            <BinaryBarChart
              name={attrName}
              values={selectedValues as number[]}
              allValues={values as number[]}
              showSelectionOnly={false}
            />
          ) : isHighCardinality ? (
            <HighCardinalityView
              name={attrName}
              values={selectedValues}
              allValues={values}
              showSelectionOnly={false}
            />
          ) : (
            <CategoricalBarChart
              name={attrName}
              values={selectedValues}
              allValues={values}
              showSelectionOnly={false}
            />
          )}
        </div>
      );
    }
  );

  return (
    <div className="selection-view" onClick={handleBackgroundClick}>
      <h3>
        Attribute Summary{" "}
        {Object.keys(selectedNodeAttributes).length > 0
          ? `(${Object.keys(selectedNodeAttributes).length} nodes selected)`
          : ""}
      </h3>
      <div className="attribute-summaries">{attributeSummaries}</div>
    </div>
  );
}

type NumericDistributionProps = {
  name: string;
  values: number[];
  allValues: number[];
  showSelectionOnly: boolean;
};

function NumericDistribution({
  name,
  values,
  allValues,
  showSelectionOnly,
}: NumericDistributionProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const hasSelection = values.length > 0;
  const [selectedNodes, setSelectedNodes] =
    useModelState<string[]>("selected_nodes");
  const [allNodeAttributes] = useModelState<NodeAttributes>(
    "all_node_attributes"
  );

  useEffect(() => {
    if (!svgRef.current || allValues.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Reduce height for more compact layout
    const width = 200;
    const height = 80;
    const margin = { top: 5, right: 10, bottom: 20, left: 25 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG group
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Determine domain based on all values
    const allMin = d3.min([...values, ...allValues]) || 0;
    const allMax = d3.max([...values, ...allValues]) || 1;

    // Create scales
    const xScale = d3
      .scaleLinear()
      .domain([allMin, allMax])
      .range([0, innerWidth])
      .nice();

    // Check if selected values are all the same
    const uniqueValues = new Set(values);
    const isConstant = uniqueValues.size === 1;
    const constantValue = isConstant ? values[0] : null;

    // Generate kernel density estimation for all values
    if (allValues.length > 0) {
      const kde = kernelDensityEstimator(
        kernelEpanechnikov(7),
        xScale.ticks(40)
      );

      const allDensity = kde(allValues);

      // Create y scale for the density
      const yScale = d3
        .scaleLinear()
        .domain([0, d3.max(allDensity, (d) => d[1]) || 0.1])
        .range([innerHeight, 0]);

      // Create the line for all values
      const line = d3
        .line<[number, number]>()
        .curve(d3.curveBasis)
        .x((d) => xScale(d[0]))
        .y((d) => yScale(d[1]));

      // Draw the background line (for all nodes)
      g.append("path")
        .datum(allDensity)
        .attr("fill", "none")
        .attr("stroke", "#4299e1") // Blue to match nodes
        .attr("stroke-width", 1.5)
        .attr("stroke-linejoin", "round")
        .attr("d", line);

      // Fill the area under the curve for all nodes
      g.append("path")
        .datum(allDensity)
        .attr("fill", "rgba(66, 153, 225, 0.3)") // Blue to match nodes
        .attr("stroke", "none")
        .attr(
          "d",
          d3
            .area<[number, number]>()
            .curve(d3.curveBasis)
            .x((d) => xScale(d[0]))
            .y0(innerHeight)
            .y1((d) => yScale(d[1]))
        );

      // Only add selection data if we have a selection
      if (hasSelection) {
        if (!isConstant && values.length > 1) {
          const selectionDensity = kde(values);

          // FIX THE SCALING - instead of scaling up, we normalize by density
          // This ensures the area under the curve represents the proportion of data selected
          const selectionArea = values.length / allValues.length;

          // Apply the scaling to make area proportional to data selected
          const normalizedSelectionDensity = selectionDensity.map((d) => [
            d[0],
            d[1] * selectionArea,
          ]);

          // Draw the selection line
          g.append("path")
            .datum(normalizedSelectionDensity)
            .attr("fill", "none")
            .attr("stroke", "#F56565") // Red stroke for selection
            .attr("stroke-width", 1.5)
            .attr("stroke-linejoin", "round")
            .attr("d", line);

          // Fill the area under the curve for selection
          g.append("path")
            .datum(normalizedSelectionDensity)
            .attr("fill", "rgba(245, 101, 101, 0.3)") // Red fill for selection
            .attr("stroke", "none")
            .attr(
              "d",
              d3
                .area<[number, number]>()
                .curve(d3.curveBasis)
                .x((d) => xScale(d[0]))
                .y0(innerHeight)
                .y1((d) => yScale(d[1]))
            );
        } else if (isConstant) {
          // For constant selected values, show a vertical line
          const constX = xScale(constantValue);

          // Draw a vertical line for the constant value
          g.append("line")
            .attr("x1", constX)
            .attr("y1", innerHeight)
            .attr("x2", constX)
            .attr("y2", 0)
            .attr("stroke", "#F56565") // Red for selection
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "3,3");

          // Draw a circle at the value point
          g.append("circle")
            .attr("cx", constX)
            .attr("cy", innerHeight / 2)
            .attr("r", 5)
            .attr("fill", "#F56565"); // Red for selection
        }
      }
    } else if (isConstant) {
      // Fallback if allValues is empty but we have constant selected values
      const constX = xScale(constantValue);

      g.append("line")
        .attr("x1", constX)
        .attr("y1", innerHeight)
        .attr("x2", constX)
        .attr("y2", 0)
        .attr("stroke", "#F56565")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "3,3");

      g.append("circle")
        .attr("cx", constX)
        .attr("cy", innerHeight / 2)
        .attr("r", 5)
        .attr("fill", "#F56565");
    }

    // Add x-axis
    const xAxis = d3.axisBottom(xScale).ticks(5);
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll("text")
      .style("font-size", "8px");

    // Add stats
    const allMean = d3.mean(allValues) || 0;
    let statText = `μ: ${allMean.toFixed(2)}`;

    if (hasSelection) {
      const mean = d3.mean(values) || 0;
      statText = `μ: ${mean.toFixed(2)} (${Math.round(
        (values.length / allValues.length) * 100
      )}%)`;
    }

    svg
      .append("text")
      .attr("x", width - 5)
      .attr("y", 10)
      .attr("text-anchor", "end")
      .style("font-size", "9px")
      .text(statText);

    // Add a selection area for brush interaction
    const brush = d3
      .brushX()
      .extent([
        [0, 0],
        [innerWidth, innerHeight],
      ])
      .on("end", brushEnded);

    // Fix: Make the overlay completely transparent
    const brushGroup = g.append("g").attr("class", "brush").call(brush);

    brushGroup.select(".overlay").attr("fill", "transparent");

    brushGroup
      .select(".selection")
      .attr("fill", "rgba(245, 101, 101, 0.05)") // Almost transparent red
      .attr("stroke", "#F56565")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "2,2"); // Dotted outline instead

    brushGroup
      .selectAll(".handle")
      .attr("stroke", "#F56565")
      .attr("stroke-width", 1)
      .attr("fill", "transparent");

    const brushText = g
      .append("text")
      .attr("class", "brush-extent")
      .attr("text-anchor", "middle")
      .attr("y", innerHeight / 2)
      .attr("fill", "#F56565")
      .attr("font-size", "9px")
      .attr("font-weight", "bold")
      .attr("pointer-events", "none") // Make sure it doesn't interfere with brushing
      .style("display", "none");

    // Fix brush event handlers with proper types
    brush
      .on("brush", (event: d3.D3BrushEvent<unknown>) => {
        if (!event.selection) return;

        const [x0, x1] = (event.selection as [number, number]).map(
          xScale.invert
        );
        const mid = (event.selection[0] + event.selection[1]) / 2;

        brushText
          .attr("x", mid)
          .text(`${x0.toFixed(1)} - ${x1.toFixed(1)}`)
          .style("display", null);
      })
      .on("end", (event: d3.D3BrushEvent<unknown>) => {
        brushText.style("display", "none");

        if (!event.selection) return; // No selection

        const [x0, x1] = (event.selection as [number, number]).map(
          xScale.invert
        );

        const nodesInRange = Object.entries(allNodeAttributes)
          .filter(([nodeId, attrs]) => {
            const value = Number(attrs[name]);
            return !isNaN(value) && value >= x0 && value <= x1;
          })
          .map(([nodeId]) => nodeId);

        if (event.sourceEvent && event.sourceEvent.shiftKey) {
          setSelectedNodes([...selectedNodes, ...nodesInRange]);
        } else {
          setSelectedNodes(nodesInRange);
        }

        if (event.sourceEvent && event.sourceEvent.type === "dblclick") {
          g.select(".brush").call(brush.move, null);
        }
      });

    function brushEnded(event: d3.D3BrushEvent<unknown>) {
      brushText.style("display", "none");

      if (!event.selection) return; // No selection

      const [x0, x1] = (event.selection as [number, number]).map(xScale.invert);

      const nodesInRange = Object.entries(allNodeAttributes)
        .filter(([nodeId, attrs]) => {
          const value = Number(attrs[name]);
          return !isNaN(value) && value >= x0 && value <= x1;
        })
        .map(([nodeId]) => nodeId);

      if (event.sourceEvent && event.sourceEvent.shiftKey) {
        setSelectedNodes([...selectedNodes, ...nodesInRange]);
      } else {
        setSelectedNodes(nodesInRange);
      }

      if (event.sourceEvent && event.sourceEvent.type === "dblclick") {
        g.select(".brush").call(brush.move, null);
      }
    }
  }, [values, allValues, name, allNodeAttributes, setSelectedNodes]);

  function kernelDensityEstimator(kernel: (v: number) => number, X: number[]) {
    return function (sample: number[]) {
      return X.map((x) => [x, d3.mean(sample, (v) => kernel(x - v)) || 0]);
    };
  }

  function kernelEpanechnikov(bandwidth: number) {
    return function (v: number) {
      return Math.abs((v /= bandwidth)) <= 1
        ? (0.75 * (1 - v * v)) / bandwidth
        : 0;
    };
  }

  return (
    <div className="numeric-distribution">
      <svg ref={svgRef} width="200" height="100"></svg>
    </div>
  );
}

// Updated BinaryBarChart component
type BinaryBarChartProps = {
  name: string;
  values: number[];
  allValues: number[];
  showSelectionOnly: boolean;
};

function BinaryBarChart({
  name,
  values,
  allValues,
  showSelectionOnly,
}: BinaryBarChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNodes, setSelectedNodes] =
    useModelState<string[]>("selected_nodes");
  const [allNodeAttributes] = useModelState<NodeAttributes>(
    "all_node_attributes"
  );

  useEffect(() => {
    if (!svgRef.current || allValues.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Reduce height for more compact layout
    const width = 200;
    const height = 80;
    const margin = { top: 5, right: 10, bottom: 30, left: 25 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Count occurrences for 0 and 1
    const zeroes = values.filter((v) => v === 0).length;
    const ones = values.filter((v) => v === 1).length;

    const allZeroes = allValues.filter((v) => v === 0).length;
    const allOnes = allValues.filter((v) => v === 1).length;

    // Create data array
    const data = [
      { label: "0", selected: zeroes, total: allZeroes },
      { label: "1", selected: ones, total: allOnes },
    ];

    // Create scales
    const xScale = d3
      .scaleBand()
      .domain(["0", "1"])
      .range([0, innerWidth])
      .padding(0.3);

    // Calculate percentages for the y scale
    const percentageZeroes = allZeroes / allValues.length;
    const percentageOnes = allOnes / allValues.length;
    const maxPercentage = Math.max(percentageZeroes, percentageOnes);

    const yScale = d3
      .scaleLinear()
      .domain([0, maxPercentage])
      .range([innerHeight, 0])
      .nice();

    // Create SVG group
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Draw background bars (all nodes)
    g.selectAll(".bar-all")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "bar-all")
      .attr("x", (d) => xScale(d.label) || 0)
      .attr("y", (d) => yScale(d.total / allValues.length))
      .attr("width", xScale.bandwidth())
      .attr("height", (d) => innerHeight - yScale(d.total / allValues.length))
      .attr("fill", "#4299e1") // Blue to match nodes
      .style("cursor", "pointer")
      .on(
        "click",
        (
          event: PointerEvent,
          d: { label: string; selected: number; total: number }
        ) => {
          // Convert category to number for binary comparison
          const categoryValue = Number(d.label);

          // Find nodes with this binary value
          const nodesWithValue = Object.entries(allNodeAttributes)
            .filter(([nodeId, attrs]) => Number(attrs[name]) === categoryValue)
            .map(([nodeId]) => nodeId);

          // Support shift+click for adding to selection
          if (event.shiftKey) {
            setSelectedNodes([...selectedNodes, ...nodesWithValue]);
          } else {
            setSelectedNodes(nodesWithValue);
          }

          // Stop event propagation
          event.stopPropagation();
        }
      );

    // Calculate the normalized heights for selection
    // This ensures the height represents the actual proportion
    const selectedTotal = values.length;
    const selectionRatio = selectedTotal / allValues.length;

    // Draw bars for selected nodes
    g.selectAll(".bar-selected")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", "bar-selected")
      .attr("x", (d) => xScale(d.label) || 0)
      .attr("y", (d) => {
        // If no nodes with this value are selected, don't show a bar
        if (d.selected === 0) return innerHeight;
        // Calculate the proper height to show the proportion of all nodes
        const selectionPercentage =
          (d.selected / selectedTotal) * selectionRatio;
        return yScale(selectionPercentage);
      })
      .attr("width", xScale.bandwidth())
      .attr("height", (d) => {
        if (d.selected === 0) return 0;
        const selectionPercentage =
          (d.selected / selectedTotal) * selectionRatio;
        return innerHeight - yScale(selectionPercentage);
      })
      .attr("fill", "#F56565"); // Red for selection

    // Add x-axis labels
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale));

    // Add y-axis showing percentages
    g.append("g").call(
      d3
        .axisLeft(yScale)
        .ticks(3)
        .tickFormat((d) => `${Math.round(d * 100)}%`)
    );

    // Add legend showing % of data selected
    const zeroesPercent = (zeroes / selectedTotal) * 100;
    const onesPercent = (ones / selectedTotal) * 100;

    svg
      .append("text")
      .attr("x", width - 5)
      .attr("y", 10)
      .attr("text-anchor", "end")
      .style("font-size", "9px")
      .text(`0: ${Math.round(zeroesPercent)}%, 1: ${Math.round(onesPercent)}%`);
  }, [values, allValues, name, allNodeAttributes, setSelectedNodes]);

  return (
    <div className="binary-chart">
      <svg ref={svgRef} width="200" height="100"></svg>
    </div>
  );
}

// Updated CategoricalBarChart component
type CategoricalBarChartProps = {
  name: string;
  values: any[];
  allValues: any[];
  showSelectionOnly: boolean;
};

function CategoricalBarChart({
  name,
  values,
  allValues,
  showSelectionOnly,
}: CategoricalBarChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNodes, setSelectedNodes] =
    useModelState<string[]>("selected_nodes");
  const [allNodeAttributes] = useModelState<NodeAttributes>(
    "all_node_attributes"
  );

  useEffect(() => {
    if (!svgRef.current || allValues.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Reduce height for more compact layout
    const width = 200;
    const height = 80;
    const margin = { top: 5, right: 10, bottom: 35, left: 25 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Count occurrences of each category for selected nodes
    const countMap = new Map<string, number>();
    values.forEach((val) => {
      const key = String(val);
      countMap.set(key, (countMap.get(key) || 0) + 1);
    });

    // Count occurrences for all nodes
    const allCountMap = new Map<string, number>();
    allValues.forEach((val) => {
      const key = String(val);
      allCountMap.set(key, (allCountMap.get(key) || 0) + 1);
    });

    // Get unique categories from both sets
    const allCategories = new Set([...countMap.keys(), ...allCountMap.keys()]);

    // Convert to array and sort by frequency in the selection
    const counts = Array.from(allCategories)
      .map((category) => ({
        category,
        selectionCount: countMap.get(category) || 0,
        totalCount: allCountMap.get(category) || 0,
        // Sort primarily by selection count, secondarily by total count
        sortValue:
          (countMap.get(category) || 0) * 1000 +
          (allCountMap.get(category) || 0),
      }))
      .sort((a, b) => b.sortValue - a.sortValue)
      .slice(0, 8); // Limit to top 8 categories

    // Store the bar order based on the initial sort
    const orderedCategories = counts.map((d) => d.category);

    // Create scales
    const xScale = d3
      .scaleBand()
      .domain(orderedCategories) // Use the stored order instead of calculating again
      .range([0, innerWidth])
      .padding(0.2);

    // Find max count for y scale - NORMALIZE BY PERCENTAGE
    const maxCountPercentage = Math.max(
      ...counts.map((d) => d.totalCount / allValues.length)
    );

    const yScale = d3
      .scaleLinear()
      .domain([0, maxCountPercentage])
      .range([innerHeight, 0])
      .nice();

    // Create SVG group
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Draw background bars for all nodes
    g.selectAll(".bar-all")
      .data(counts)
      .enter()
      .append("rect")
      .attr("class", "bar-all")
      .attr("x", (d) => xScale(d.category) || 0)
      .attr("y", (d) => yScale(d.totalCount / allValues.length))
      .attr("width", xScale.bandwidth())
      .attr(
        "height",
        (d) => innerHeight - yScale(d.totalCount / allValues.length)
      )
      .attr("fill", "#4299e1") // Blue to match nodes
      .style("cursor", "pointer")
      .on(
        "click",
        (
          event: PointerEvent,
          d: {
            category: string;
            selectionCount: number;
            totalCount: number;
            sortValue: number;
          }
        ) => {
          event.preventDefault(); // Prevent default behavior

          // Find nodes with this attribute value
          const nodesWithValue = Object.entries(allNodeAttributes)
            .filter(([nodeId, attrs]) => String(attrs[name]) === d.category)
            .map(([nodeId]) => nodeId);

          // Support shift+click to add to selection
          if (event.shiftKey) {
            setSelectedNodes([...selectedNodes, ...nodesWithValue]);
          } else {
            setSelectedNodes(nodesWithValue);
          }

          // Stop event propagation
          event.stopPropagation();
        }
      );

    // Calculate the selected ratio for scaling
    const selectedTotal = values.length;
    const selectionRatio = selectedTotal / allValues.length;

    // Draw bars for selected nodes - FIX THE SCALING
    g.selectAll(".bar-selected")
      .data(counts)
      .enter()
      .append("rect")
      .attr("class", "bar-selected")
      .attr("x", (d) => xScale(d.category) || 0)
      .attr("y", (d) => {
        // If no nodes with this category are selected, don't show a bar
        if (d.selectionCount === 0) return innerHeight;
        // Calculate the proper height to show the true proportion
        const selectionPercentage =
          (d.selectionCount / selectedTotal) * selectionRatio;
        return yScale(selectionPercentage);
      })
      .attr("width", xScale.bandwidth())
      .attr("height", (d) => {
        if (d.selectionCount === 0) return 0;
        const selectionPercentage =
          (d.selectionCount / selectedTotal) * selectionRatio;
        return innerHeight - yScale(selectionPercentage);
      })
      .attr("fill", "#F56565"); // Red for selection

    // Add axes - CHANGED Y AXIS TO SHOW PERCENTAGES
    const xAxis = d3.axisBottom(xScale);
    const yAxis = d3
      .axisLeft(yScale)
      .ticks(3)
      .tickFormat((d) => `${Math.round(d * 100)}%`);

    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(xAxis)
      .selectAll("text")
      .style("font-size", "7px")
      .attr("transform", "rotate(-45)")
      .attr("text-anchor", "end")
      .attr("dx", "-0.6em")
      .attr("dy", "0.15em");

    g.append("g").call(yAxis).selectAll("text").style("font-size", "8px");

    // Add legend or count info
    svg
      .append("text")
      .attr("x", width - 5)
      .attr("y", 10)
      .attr("text-anchor", "end")
      .style("font-size", "9px")
      .text(
        `${Math.round((values.length / allValues.length) * 100)}% selected`
      );
  }, [values, allValues, name, allNodeAttributes, setSelectedNodes]);

  return (
    <div className="categorical-chart">
      <svg ref={svgRef} width="200" height="120"></svg>
    </div>
  );
}

// Add after the other chart components

type HighCardinalityViewProps = {
  name: string;
  values: any[];
  allValues: any[];
  showSelectionOnly: boolean;
};

function HighCardinalityView({
  name,
  values,
  allValues,
  showSelectionOnly,
}: HighCardinalityViewProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const uniqueAllValues = new Set(allValues);
  const uniqueSelectedValues = new Set(values);
  const [selectedNodes, setSelectedNodes] =
    useModelState<string[]>("selected_nodes");
  const [allNodeAttributes] = useModelState<NodeAttributes>(
    "all_node_attributes"
  );

  useEffect(() => {
    if (!svgRef.current || allValues.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Configuration
    const width = 200;
    const height = 80;
    const margin = { top: 5, right: 10, bottom: 20, left: 25 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create SVG group
    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Count stats about unique values
    const totalUniqueCount = uniqueAllValues.size;
    const selectedUniqueCount = uniqueSelectedValues.size;
    const overlapCount = [...uniqueSelectedValues].filter((v) =>
      uniqueAllValues.has(v)
    ).length;

    // Create scales for the breakdown
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerWidth]);

    // Add a text summary instead of a chart
    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .text(`${totalUniqueCount} unique values`);

    if (values.length > 0) {
      g.append("text")
        .attr("x", innerWidth / 2)
        .attr("y", 35)
        .attr("text-anchor", "middle")
        .attr("font-size", "10px")
        .attr("fill", "#F56565")
        .text(`${selectedUniqueCount} unique in selection`);

      // How many are shared vs unique to selection
      g.append("text")
        .attr("x", innerWidth / 2)
        .attr("y", 50)
        .attr("text-anchor", "middle")
        .attr("font-size", "10px")
        .text(
          `${((values.length / allValues.length) * 100).toFixed(
            1
          )}% of data selected`
        );
    }

    // Create a simple cardinality indicator bar
    const cardinalityRect = g
      .append("rect")
      .attr("x", 0)
      .attr("y", innerHeight - 10)
      .attr("width", innerWidth)
      .attr("height", 8)
      .attr("fill", "#E2E8F0")
      .attr("rx", 3);

    // Indicator showing relative size of selection
    if (values.length > 0) {
      const selectionWidth = Math.max(
        3, // Minimum width for visibility
        (selectedUniqueCount / totalUniqueCount) * innerWidth
      );

      g.append("rect")
        .attr("x", 0)
        .attr("y", innerHeight - 10)
        .attr("width", selectionWidth)
        .attr("height", 8)
        .attr("fill", "#F56565")
        .attr("rx", 3);
    }

    // Add a caption
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height - 5)
      .attr("text-anchor", "middle")
      .attr("font-size", "8px")
      .text("High cardinality attribute (summary view)");

    // Add a "select top 10 most common values" button
    g.append("rect")
      .attr("x", 20)
      .attr("y", innerHeight - 30)
      .attr("width", innerWidth - 40)
      .attr("height", 20)
      .attr("rx", 3)
      .attr("fill", "#4299e1")
      .style("cursor", "pointer")
      .on("click", (event: PointerEvent) => {
        // Count occurrences of each value
        const valueCounts = new Map<any, number>();
        Object.entries(allNodeAttributes).forEach(([nodeId, attrs]) => {
          const val = attrs[name];
          valueCounts.set(val, (valueCounts.get(val) || 0) + 1);
        });

        // Get top 10 most common values
        const topValues = [...valueCounts.entries()]
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10)
          .map(([val]) => val);

        // Select nodes with these values
        const nodesToSelect = Object.entries(allNodeAttributes)
          .filter(([nodeId, attrs]) => topValues.includes(attrs[name]))
          .map(([nodeId]) => nodeId);

        setSelectedNodes(nodesToSelect);
      });

    g.append("text")
      .attr("x", innerWidth / 2)
      .attr("y", innerHeight - 17)
      .attr("text-anchor", "middle")
      .attr("fill", "white")
      .attr("font-size", "9px")
      .text("Select top 10 most common values")
      .style("pointer-events", "none");
  }, [values, allValues, name, allNodeAttributes, setSelectedNodes]);

  return (
    <div className="high-cardinality-view">
      <svg ref={svgRef} width="200" height="100"></svg>
    </div>
  );
}
