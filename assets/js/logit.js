async function drawScatter() {

  // Access data
  let dataset = await d3.csv("../assets/logit_lin_sep.csv")

  dataset.forEach(d => {
    d["x1"] = +d["x1"]
    d["y1"] = +d["y1"]
    d["class"] = +d["class"]
  })

  //Set data constants
  const xAccessor = d => d["x1"]
  const yAccessor = d => d["y1"]
  const colorAccessor = d => d["class"]
  const x2Accessor = d => d["x2"]
  const y2Accessor = d => d["y2"]
  const x3Accessor = d => d["x3"]
  const y3Accessor = d => d["y3"]

  // Create chart dimensions

  const width = d3.min([
    window.innerWidth * 0.9,
    window.innerHeight * 0.9,
  ])
  let dimensions = {
    width: width,
    height: width,
    margin: {
      top: 10,
      right: 10,
      bottom: 50,
      left: 50,
    },
  }
  dimensions.boundedWidth = dimensions.width
    - dimensions.margin.left
    - dimensions.margin.right
  dimensions.boundedHeight = dimensions.height
    - dimensions.margin.top
    - dimensions.margin.bottom

  // Draw canvas

  const wrapper = d3.select("#wrapper")
    .append("svg")
    .attr("width", dimensions.width)
    .attr("height", dimensions.height)

  const bounds = wrapper.append("g")
    .style("transform", `translate(${
      dimensions.margin.left
      }px, ${
      dimensions.margin.top
      }px)`)

  // Create scales

  xScale = d3.scaleLinear()
    .domain(d3.extent(dataset, xAccessor))
    .range([0, dimensions.boundedWidth])
    .nice()

  yScale = d3.scaleLinear()
    .domain(d3.extent(dataset, yAccessor))
    .range([dimensions.boundedHeight, 0])
    .nice()

  const colorScale = d3.scaleQuantize()
    .domain(d3.extent(dataset, colorAccessor))
    .range(["#8C54D0", "#19B092"])

  // Draw data
  dots = bounds.selectAll("circle")
    .data(dataset)
    .enter().append("circle")
    .attr("cx", d => xScale(xAccessor(d)))
    .attr("cy", d => yScale(yAccessor(d)))
    .attr("r", 6)
    .attr("fill", "#F5F5F2")//d => colorScale(colorAccessor(d)))
    .attr('stroke', d => colorScale(colorAccessor(d)))
    .attr('stroke-width', 4)


  //Interaction mean line
  // let drag = d3.drag()
  //   .on("start", dragstarted)
  //   .on('drag', dragged)
  //   .on('end', dragended)

  //Draw mean line
  const mean = d3.mean(dataset, yAccessor)
  const meanLine = bounds.append("line")
    .attr("x1", -1)
    .attr("x2", dimensions.boundedWidth)
    .attr("y1", yScale(mean))
    .attr("y2", yScale(mean))
    .attr("stroke", "#CF366C")
    .attr("stroke-width", 5)
    .attr("opacity", 0.6)
    //.call(drag)

  //Set up transitions for the buttons
  d3.select("#caseTwo").on("click", function () {
    dots
      .transition()
      .duration(1200)
      .ease(d3.easeCubicIn)
      .attr("cx", d => xScale(x2Accessor(d)))
      .attr("cy", d => yScale(y2Accessor(d)))
  })

  d3.select("#caseOne").on("click", function () {
    dots
      .transition()
      .duration(1200)
      .ease(d3.easeCubicIn)
      .attr("cx", d => xScale(xAccessor(d)))
      .attr("cy", d => yScale(yAccessor(d)))
  })

  d3.select("#caseThree").on("click", function () {
    dots
      .transition()
      .duration(1200)
      .ease(d3.easeCubicIn)
      .attr("cx", d => xScale(x3Accessor(d)))
      .attr("cy", d => yScale(y3Accessor(d)))
  })

  // d3.select("#reset").on("click", function () {
  //   meanLine
  //     .transition()
  //     .duration(1200)
  //     .ease(d3.easeCubicIn)
  //     .attr("x1", -1)
  //     .attr("x2", dimensions.boundedWidth)
  //     .attr("y1", yScale(mean))
  //     .attr("y2", yScale(mean))
  // })

  // Draw peripherals

  xAxisGenerator = d3.axisBottom()
    .scale(xScale)

  xAxis = bounds.append("g")
    .call(xAxisGenerator)
    .style("transform", `translateY(${dimensions.boundedHeight}px)`)

  xAxisLabel = xAxis.append("text")
    .attr("x", dimensions.boundedWidth / 2)
    .attr("y", dimensions.margin.bottom - 10)
    .attr("fill", "black")
    .style("font-size", "1.4em")
    .html("Feature A")

  yAxisGenerator = d3.axisLeft()
    .scale(yScale)

  yAxis = bounds.append("g")
    .call(yAxisGenerator)

  yAxisLabel = yAxis.append("text")
    .attr("x", -dimensions.boundedHeight / 2)
    .attr("y", -dimensions.margin.left + 10)
    .attr("fill", "black")
    .style("font-size", "1.4em")
    .text("Feature B")
    .style("transform", "rotate(-90deg)")
    .style("text-anchor", "middle")

  //Interactions
  // function dragstarted(d) {
  //   d3.select(this).raise().classed('active', true)
  // }

  // function dragged(d) {
  //   dx = d3.event.dx
  //   dy = d3.event.dy
  //   x1New = parseFloat(d3.select(this).attr('x1')) + dx
  //   y1New = parseFloat(d3.select(this).attr('y1')) + dy
  //   x2New = parseFloat(d3.select(this).attr('x2')) + dx
  //   y2New = parseFloat(d3.select(this).attr('y2')) + dy
  //   d3.selectAll(this)
  //     .attr("x1", d3.event.x1 - 1)
  //     .attr("y1", d3.event.y1 + yScale(mean))
  //     .attr("x2", d3.event.x2 + dimensions.boundedWidth)
  //     .attr("y2", d3.event.y2 + yScale(mean))
  // }

  // function dragended(d) {
  //   d3.select(this).classed('active', false)
  // }
}
drawScatter()
