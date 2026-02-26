/**
 * ChipStack Research Visualization Charts
 * 图表可视化脚本 - 修复重复渲染问题
 */

// Color scheme
const colors = {
  primary: '#00b4d8',
  secondary: '#7b2cbf',
  accent: '#48cae4',
  gradient: ['#00b4d8', '#48cae4', '#90e0ef', '#7b2cbf'],
  bars: ['#00b4d8', '#48cae4', '#90e0ef', '#0077b6', '#03045e']
};

// Track initialized charts to prevent duplicates
const initializedCharts = new Set();

// Academic Citation Chart Data
const academicData = {
  labels: ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025'],
  datasets: [{
    label: 'Citations',
    data: [50000, 120000, 250000, 450000, 650000, 820000, 950000, 1044867],
    borderColor: colors.primary,
    backgroundColor: 'rgba(0, 180, 216, 0.1)',
    fill: true,
    tension: 0.4
  }]
};

// Technical Skills Radar Data
const skillsData = {
  labels: [
    'AI/ML',
    'Hardware Design',
    'Verification',
    'System Architecture',
    'Low Power',
    'Software'
  ],
  datasets: [
    {
      label: 'Kartik Hegde',
      data: [95, 85, 80, 90, 75, 85],
      borderColor: colors.primary,
      backgroundColor: 'rgba(0, 180, 216, 0.2)',
      pointBackgroundColor: colors.primary
    },
    {
      label: 'Hamid Shojaei',
      data: [90, 95, 90, 85, 85, 80],
      borderColor: colors.secondary,
      backgroundColor: 'rgba(123, 44, 191, 0.2)',
      pointBackgroundColor: colors.secondary
    }
  ]
};

// Customer Results Data
const customerData = {
  labels: ['Altera', 'Tenstorrent', 'NVIDIA', 'Qualcomm'],
  datasets: [{
    label: 'Efficiency Improvement (x)',
    data: [10, 4, 3, 2.5],
    backgroundColor: colors.bars,
    borderRadius: 8
  }]
};

// Initialize Charts when DOM is loaded - ONLY ONCE
document.addEventListener('DOMContentLoaded', function() {
  // Only initialize overview charts on first load
  initCustomerChart();
});

// Academic Citation Trend Chart
function initAcademicChart() {
  const container = document.getElementById('academic-chart');
  if (!container) return;

  // Prevent duplicate rendering
  if (initializedCharts.has('academic-chart')) return;
  initializedCharts.add('academic-chart');

  // Clear existing content
  container.innerHTML = '';

  const width = container.offsetWidth || 600;
  const height = 300;
  const padding = { top: 20, right: 30, bottom: 40, left: 80 };

  const svg = createSVG(width, height);
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Scale data
  const maxVal = Math.max(...academicData.datasets[0].data);
  const xScale = chartWidth / (academicData.labels.length - 1);
  const yScale = chartHeight / maxVal;

  // Gradient definition - must be added first
  const defs = createSVGElement('defs');
  const gradient = createSVGElement('linearGradient', {
    id: 'areaGradient',
    x1: '0%',
    y1: '0%',
    x2: '0%',
    y2: '100%'
  });
  const stop1 = createSVGElement('stop', { offset: '0%', 'stop-color': colors.primary, 'stop-opacity': '0.5' });
  const stop2 = createSVGElement('stop', { offset: '100%', 'stop-color': colors.primary, 'stop-opacity': '0' });
  gradient.appendChild(stop1);
  gradient.appendChild(stop2);
  defs.appendChild(gradient);
  svg.appendChild(defs);

  // Draw grid lines
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    const line = createSVGElement('line', {
      x1: padding.left,
      y1: y,
      x2: width - padding.right,
      y2: y,
      stroke: 'rgba(255,255,255,0.1)',
      'stroke-dasharray': '5,5'
    });
    svg.appendChild(line);

    // Y-axis labels
    const value = Math.round(maxVal - (maxVal / 5) * i);
    const text = createSVGElement('text', {
      x: padding.left - 10,
      y: y + 4,
      'text-anchor': 'end',
      fill: '#a0aec0',
      'font-size': '12'
    });
    text.textContent = formatNumber(value);
    svg.appendChild(text);
  }

  // Draw line
  let pathD = '';
  const points = [];
  academicData.datasets[0].data.forEach((val, i) => {
    const x = padding.left + i * xScale;
    const y = padding.top + chartHeight - (val * yScale);
    points.push({ x, y });
    if (i === 0) {
      pathD += `M ${x} ${y}`;
    } else {
      pathD += ` L ${x} ${y}`;
    }
  });

  // Area fill
  const areaPath = createSVGElement('path', {
    d: pathD + ` L ${points[points.length-1].x} ${padding.top + chartHeight} L ${padding.left} ${padding.top + chartHeight} Z`,
    fill: 'url(#areaGradient)',
    opacity: '0.3'
  });
  svg.appendChild(areaPath);

  // Line
  const linePath = createSVGElement('path', {
    d: pathD,
    fill: 'none',
    stroke: colors.primary,
    'stroke-width': '3',
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round'
  });
  svg.appendChild(linePath);

  // Points and X-axis labels
  points.forEach((point, i) => {
    const circle = createSVGElement('circle', {
      cx: point.x,
      cy: point.y,
      r: '6',
      fill: colors.primary,
      stroke: '#0a0e27',
      'stroke-width': '2'
    });
    svg.appendChild(circle);

    // X-axis labels
    const text = createSVGElement('text', {
      x: point.x,
      y: height - 10,
      'text-anchor': 'middle',
      fill: '#a0aec0',
      'font-size': '12'
    });
    text.textContent = academicData.labels[i];
    svg.appendChild(text);
  });

  container.appendChild(svg);
}

// Skills Radar Chart
function initSkillsRadar() {
  const container = document.getElementById('skills-radar');
  if (!container) return;

  // Prevent duplicate rendering
  if (initializedCharts.has('skills-radar')) return;
  initializedCharts.add('skills-radar');

  // Clear existing content
  container.innerHTML = '';

  const width = container.offsetWidth || 400;
  const height = 400;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) / 2 - 50;
  const levels = 5;
  const numSkills = skillsData.labels.length;
  const angleStep = (Math.PI * 2) / numSkills;

  const svg = createSVG(width, height);

  // Draw grid circles
  for (let i = 1; i <= levels; i++) {
    const r = (radius / levels) * i;
    const circle = createSVGElement('circle', {
      cx: centerX,
      cy: centerY,
      r: r,
      fill: 'none',
      stroke: 'rgba(255,255,255,0.1)',
      'stroke-dasharray': '5,5'
    });
    svg.appendChild(circle);
  }

  // Draw axes and labels
  skillsData.labels.forEach((label, i) => {
    const angle = angleStep * i - Math.PI / 2;
    const x = centerX + Math.cos(angle) * radius;
    const y = centerY + Math.sin(angle) * radius;

    // Axis line
    const line = createSVGElement('line', {
      x1: centerX,
      y1: centerY,
      x2: x,
      y2: y,
      stroke: 'rgba(255,255,255,0.2)'
    });
    svg.appendChild(line);

    // Label
    const labelX = centerX + Math.cos(angle) * (radius + 25);
    const labelY = centerY + Math.sin(angle) * (radius + 25);
    const text = createSVGElement('text', {
      x: labelX,
      y: labelY,
      'text-anchor': 'middle',
      'dominant-baseline': 'middle',
      fill: '#a0aec0',
      'font-size': '12'
    });
    text.textContent = label;
    svg.appendChild(text);
  });

  // Draw data polygons
  skillsData.datasets.forEach((dataset, datasetIndex) => {
    let pathD = '';
    dataset.data.forEach((value, i) => {
      const angle = angleStep * i - Math.PI / 2;
      const r = (value / 100) * radius;
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;
      if (i === 0) {
        pathD += `M ${x} ${y}`;
      } else {
        pathD += ` L ${x} ${y}`;
      }
    });
    pathD += ' Z';

    // Fill
    const fill = createSVGElement('path', {
      d: pathD,
      fill: datasetIndex === 0 ? 'rgba(0, 180, 216, 0.2)' : 'rgba(123, 44, 191, 0.2)',
      stroke: dataset.borderColor,
      'stroke-width': '2'
    });
    svg.appendChild(fill);

    // Points
    dataset.data.forEach((value, i) => {
      const angle = angleStep * i - Math.PI / 2;
      const r = (value / 100) * radius;
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;
      const circle = createSVGElement('circle', {
        cx: x,
        cy: y,
        r: '5',
        fill: dataset.borderColor,
        stroke: '#0a0e27',
        'stroke-width': '2'
      });
      svg.appendChild(circle);
    });
  });

  container.appendChild(svg);
}

// Customer Results Bar Chart
function initCustomerChart() {
  const container = document.getElementById('customer-chart');
  if (!container) return;

  // Prevent duplicate rendering
  if (initializedCharts.has('customer-chart')) return;
  initializedCharts.add('customer-chart');

  // Clear existing content
  container.innerHTML = '';

  const width = container.offsetWidth || 500;
  const height = 250;
  const padding = { top: 20, right: 30, bottom: 50, left: 60 };

  const svg = createSVG(width, height);
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const maxVal = Math.max(...customerData.datasets[0].data);
  const barWidth = chartWidth / customerData.labels.length - 20;

  // Draw grid lines
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    const line = createSVGElement('line', {
      x1: padding.left,
      y1: y,
      x2: width - padding.right,
      y2: y,
      stroke: 'rgba(255,255,255,0.1)',
      'stroke-dasharray': '5,5'
    });
    svg.appendChild(line);

    const value = Math.round(maxVal - (maxVal / 5) * i);
    const text = createSVGElement('text', {
      x: padding.left - 10,
      y: y + 4,
      'text-anchor': 'end',
      fill: '#a0aec0',
      'font-size': '12'
    });
    text.textContent = value + 'x';
    svg.appendChild(text);
  }

  // Draw bars
  customerData.datasets[0].data.forEach((val, i) => {
    const barHeight = (val / maxVal) * chartHeight;
    const x = padding.left + i * (chartWidth / customerData.labels.length) + 10;
    const y = padding.top + chartHeight - barHeight;

    // Bar
    const rect = createSVGElement('rect', {
      x: x,
      y: y,
      width: barWidth,
      height: barHeight,
      fill: customerData.datasets[0].backgroundColor[i],
      rx: '8',
      ry: '8'
    });
    svg.appendChild(rect);

    // Value label
    const valueText = createSVGElement('text', {
      x: x + barWidth / 2,
      y: y - 10,
      'text-anchor': 'middle',
      fill: '#48cae4',
      'font-size': '14',
      'font-weight': 'bold'
    });
    valueText.textContent = val + 'x';
    svg.appendChild(valueText);

    // X-axis label
    const label = createSVGElement('text', {
      x: x + barWidth / 2,
      y: height - 15,
      'text-anchor': 'middle',
      fill: '#a0aec0',
      'font-size': '12'
    });
    label.textContent = customerData.labels[i];
    svg.appendChild(label);
  });

  container.appendChild(svg);
}

// Animated Counters - with duplicate check
function initAnimatedCounters() {
  const counters = document.querySelectorAll('[data-counter]');
  counters.forEach(counter => {
    // Skip if already animated
    if (counter.dataset.animated === 'true') return;
    counter.dataset.animated = 'true';

    const target = parseInt(counter.getAttribute('data-counter'));
    const duration = 2000;
    const start = 0;
    const startTime = performance.now();

    function updateCounter(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const current = Math.floor(start + (target - start) * easeOut);

      counter.textContent = formatNumber(current);

      if (progress < 1) {
        requestAnimationFrame(updateCounter);
      }
    }

    requestAnimationFrame(updateCounter);
  });
}

// Helper Functions
function createSVG(width, height) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  return svg;
}

function createSVGElement(tag, attributes = {}) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, value);
  }
  return element;
}

function formatNumber(num) {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  } else if (num >= 1000) {
    return (num / 1000).toFixed(0) + 'K';
  }
  return num.toString();
}

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
  @keyframes drawPath {
    to {
      stroke-dashoffset: 0;
    }
  }
  @keyframes flowPulse {
    0%, 100% { opacity: 0.5; transform: translateX(0); }
    50% { opacity: 1; transform: translateX(5px); }
  }
`;
document.head.appendChild(style);
