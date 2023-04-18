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

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);
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

const results = {
  bird: 0.1,
  cat: 0.2,
  dog: 0.2,
  fish: 0.3,
  horse: 0.1,
  mouse: 0.6,
  rabbit: 0.7,
};
const labels = Object.keys(results);
const values = Object.values(results);
values.sort(function (a, b) {
  return b - a;
});
console.log(values);
export const data = {
  labels,
  datasets: [
    {
      label: "Dataset 1",
      data: values,
      borderColor: "rgb(53, 162, 235)",
      backgroundColor: "rgb(53, 162, 235)",
    } /*
    {
      label: "Dataset 2",
      data: [20, 30, 20, 10, 10, 5, 0],
      borderColor: "rgb(53, 162, 235)",
      backgroundColor: "rgba(53, 162, 235, 0.5)",
    },*/,
  ],
};

export default function Example() {
  return <Bar options={options} data={data} />;
}
