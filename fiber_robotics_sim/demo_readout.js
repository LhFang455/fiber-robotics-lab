/* Pure readout formatting shared by every explanatory scene. */
function demoSignalReadout(value, maximum, scientific, pendingText) {
  if (!Number.isFinite(value)) {
    return {text: pendingText || '缺测 / 无有效观测', width: 0, color: '#737e89'};
  }
  const scale = Number.isFinite(maximum) && maximum > 0 ? maximum : 1;
  return {
    text: scientific ? value.toExponential(3) : value.toFixed(4),
    width: Math.max(0, Math.min(100, Math.abs(value) / scale * 100)),
    color: value < 0 ? '#94baf4' : '#67dcc6'
  };
}

function demoViewKey(code, yaw, elevation, defaults) {
  if (code === 'Home') return {yaw: defaults.yaw, elevation: defaults.elevation};
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(code)) return null;
  return {
    yaw: yaw + (code === 'ArrowLeft' ? .12 : code === 'ArrowRight' ? -.12 : 0),
    elevation: Math.max(.12, Math.min(1.42, elevation + (code === 'ArrowUp' ? .1 : code === 'ArrowDown' ? -.1 : 0)))
  };
}

function demoSetLayerVisibility(objects, visible) {
  objects.forEach(object => { object.visible = visible; });
}
