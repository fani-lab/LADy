import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import React, { useState, useEffect } from "react";
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);
const plugins = [
  {
    afterDraw: function (chart) {
      console.log(chart);
      if (chart.data.datasets[0].data.length < 1) {
        let ctx = chart.ctx;
        let width = chart.width;
        let height = chart.height;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "30px Arial";
        ctx.fillText("No data to display", width / 2, height / 2);
        ctx.restore();
      }
    },
  },
];
export const options = {
  indexAxis: "y",
  elements: {
    bar: {
      borderWidth: 2,
    },
  },
  responsive: true,
  plugins: {
    legend: {
      position: "right",
    },
    title: {
      display: true,
      text: "Aspect Results",
    },
  },
};

export default function Example(props) {
  console.log("output", props.output);
  const [chartData, setChartData] = useState({
    datasets: [],
  });

  return <Bar options={options} data={chartData} plugins={plugins} />;
}
