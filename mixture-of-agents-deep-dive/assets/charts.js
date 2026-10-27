(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  var axisCommon = {
    axisLine: { lineStyle: { color: rule } },
    axisTick: { show: false },
    axisLabel: { color: muted, fontSize: 11 },
    splitLine: { lineStyle: { color: rule, type: 'dashed' } }
  };

  // ---------- Chart: benchmark split ----------
  var elBench = document.getElementById('chart-bench');
  if (elBench && window.echarts) {
    var bench = echarts.init(elBench, null, { renderer: 'svg' });
    bench.setOption({
      animation: false,
      tooltip: {
        trigger: 'axis',
        appendToBody: true,
        axisPointer: { type: 'shadow' },
        formatter: function (params) {
          var s = params[0].axisValue.replace('\n', ' ') + '<br/>';
          params.forEach(function (p) {
            s += p.marker + p.seriesName + ': <b>' + p.value + '</b><br/>';
          });
          return s;
        }
      },
      legend: {
        top: 0,
        right: 0,
        itemWidth: 10,
        itemHeight: 10,
        textStyle: { color: muted, fontSize: 11 }
      },
      grid: { left: 44, right: 18, top: 52, bottom: 56 },
      xAxis: {
        type: 'category',
        data: ['MoA\n3层/6代理', 'MoA w/ GPT-4o', 'MoA-Lite\n2层', 'GPT-4o', 'GPT-4 Turbo'],
        axisLine: axisCommon.axisLine,
        axisTick: axisCommon.axisTick,
        axisLabel: { color: ink, fontSize: 10.5, interval: 0, lineHeight: 14 }
      },
      yAxis: {
        type: 'value',
        min: 40,
        max: 100,
        name: '胜率 (%)',
        nameTextStyle: { color: muted, fontSize: 10, padding: [0, 0, 6, 0] },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: axisCommon.axisLabel,
        splitLine: axisCommon.splitLine
      },
      series: [
        {
          name: 'AlpacaEval 2.0 LC',
          type: 'bar',
          data: [65.1, 65.7, 59.3, 57.5, 55.0],
          barMaxWidth: 26,
          itemStyle: { color: accent, borderRadius: [2, 2, 0, 0] },
          label: { show: true, position: 'top', color: ink, fontSize: 10.5, formatter: '{c}' }
        },
        {
          name: 'Arena-Hard',
          type: 'bar',
          data: [77.4, 90.3, 71.3, 79.2, 82.0],
          barMaxWidth: 26,
          itemStyle: { color: accent2, borderRadius: [2, 2, 0, 0] },
          label: { show: true, position: 'top', color: ink, fontSize: 10.5, formatter: '{c}' }
        }
      ]
    });
    window.addEventListener('resize', function () { bench.resize(); });
  }

  // ---------- Chart: model calls ----------
  var elCalls = document.getElementById('chart-calls');
  if (elCalls && window.echarts) {
    var calls = echarts.init(elCalls, null, { renderer: 'svg' });
    calls.setOption({
      animation: false,
      tooltip: {
        trigger: 'axis',
        appendToBody: true,
        axisPointer: { type: 'shadow' },
        formatter: function (params) {
          return params[0].name + '<br/>模型调用次数: <b>' + params[0].value + '</b>';
        }
      },
      grid: { left: 80, right: 48, top: 24, bottom: 36 },
      xAxis: {
        type: 'value',
        name: '次 / 请求',
        nameTextStyle: { color: muted, fontSize: 10, padding: [0, 0, 0, 20] },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: axisCommon.axisLabel,
        splitLine: axisCommon.splitLine,
        min: 0,
        max: 21
      },
      yAxis: {
        type: 'category',
        data: ['单模型', 'Self-Consistency (k=6)', 'MoA-Lite (2层×6+1)', 'MoA (3层×6+1)'],
        axisLine: { lineStyle: { color: rule } },
        axisTick: { show: false },
        axisLabel: { color: ink, fontSize: 11 }
      },
      series: [{
        type: 'bar',
        data: [
          { value: 1, itemStyle: { color: muted } },
          { value: 6, itemStyle: { color: accent2 } },
          { value: 13, itemStyle: { color: accent } },
          { value: 19, itemStyle: { color: accent, opacity: 0.75 } }
        ],
        barMaxWidth: 22,
        itemStyle: { borderRadius: [0, 2, 2, 0] },
        label: { show: true, position: 'right', color: ink, fontSize: 11, formatter: '{c}×' }
      }]
    });
    window.addEventListener('resize', function () { calls.resize(); });
  }

  // ---------- Mermaid ----------
  if (window.mermaid) {
    mermaid.initialize({
      startOnLoad: false,
      theme: 'neutral',
      securityLevel: 'loose',
      themeVariables: {
        fontFamily: "Instrument Sans, Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif",
        fontSize: '13px',
        primaryColor: bg2,
        primaryBorderColor: rule,
        primaryTextColor: ink,
        lineColor: muted,
        tertiaryColor: bg2
      }
    });
    mermaid.run({ querySelector: '.mermaid' });
  }
})();
