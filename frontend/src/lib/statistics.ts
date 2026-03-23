/**
 * Calculate statistical measurements for numeric columns
 */

export interface ColumnStatistics {
  columnName: string;
  isNumeric: boolean;
  mean?: number;
  median?: number;
  mode?: number;
  range?: number;
  variance?: number;
  stdDeviation?: number;
  min?: number;
  max?: number;
  count: number;
}

/**
 * Extract numeric values from a column, filtering out null/invalid values
 */
function extractNumericValues(values: any[]): number[] {
  return values
    .map((v) => parseFloat(v))
    .filter((v) => !isNaN(v));
}

/**
 * Calculate mean (average)
 */
function calculateMean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

/**
 * Calculate median (middle value)
 */
function calculateMedian(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  
  if (sorted.length % 2 !== 0) {
    return sorted[mid];
  }
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Calculate mode (most frequent value)
 * Returns the first mode if multiple exist
 */
function calculateMode(values: number[]): number {
  if (values.length === 0) return 0;
  
  const frequency: Record<number, number> = {};
  
  for (const value of values) {
    frequency[value] = (frequency[value] || 0) + 1;
  }
  
  let mode = values[0];
  let maxFrequency = 0;
  
  for (const [value, count] of Object.entries(frequency)) {
    if (count > maxFrequency) {
      maxFrequency = count;
      mode = parseFloat(value);
    }
  }
  
  return mode;
}

/**
 * Calculate range (max - min)
 */
function calculateRange(values: number[]): number {
  if (values.length === 0) return 0;
  const min = Math.min(...values);
  const max = Math.max(...values);
  return max - min;
}

/**
 * Calculate variance
 */
function calculateVariance(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = calculateMean(values);
  const squaredDifferences = values.map((val) => Math.pow(val - mean, 2));
  return squaredDifferences.reduce((sum, val) => sum + val, 0) / values.length;
}

/**
 * Calculate standard deviation
 */
function calculateStandardDeviation(values: number[]): number {
  const variance = calculateVariance(values);
  return Math.sqrt(variance);
}

/**
 * Calculate all statistics for a given column
 */
export function calculateColumnStats(
  columnName: string,
  values: any[]
): ColumnStatistics {
  const numericValues = extractNumericValues(values);
  const isNumeric = numericValues.length > values.filter((v) => v != null && String(v).trim() !== "").length * 0.8;
  
  if (!isNumeric || numericValues.length === 0) {
    return {
      columnName,
      isNumeric: false,
      count: 0,
    };
  }
  
  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);
  
  return {
    columnName,
    isNumeric: true,
    mean: calculateMean(numericValues),
    median: calculateMedian(numericValues),
    mode: calculateMode(numericValues),
    range: calculateRange(numericValues),
    variance: calculateVariance(numericValues),
    stdDeviation: calculateStandardDeviation(numericValues),
    min,
    max,
    count: numericValues.length,
  };
}

/**
 * Calculate statistics for multiple columns
 */
export function calculateMultipleColumnStats(
  rows: any[],
  columnNames: string[]
): ColumnStatistics[] {
  return columnNames.map((colName) => {
    const values = rows.map((row) => row[colName]);
    return calculateColumnStats(colName, values);
  });
}

/**
 * Format a number for display (round to 2 decimal places)
 */
export function formatStatValue(value: number | undefined): string {
  if (value === undefined) return "—";
  if (Number.isInteger(value)) return value.toString();
  return value.toFixed(2);
}
