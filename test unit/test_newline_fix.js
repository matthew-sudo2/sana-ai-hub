#!/usr/bin/env node
/**
 * Test script to demonstrate the newline stripping bug and its fix
 */

const testMarkdown = `# Validation Report

## Summary
- **Status:** APPROVED
- **Overall Confidence:** 92.5%
- **Source:** backend/test_data.csv
- **Type:** csv

## Dimension Scores
- completeness: 100.0% (weight 25%, 8/8)
- sanity: 78.4% (weight 35%, 5/8)

## Check Details
- **PASS** [completeness] Scout output JSON present
- **PASS** [sanity] No duplicate rows`;

console.log("=".repeat(80));
console.log("VALIDATION REPORT NEWLINE BUG - BEFORE AND AFTER");
console.log("=".repeat(80));

console.log("\n📋 ORIGINAL MARKDOWN (with proper newlines):");
console.log("-".repeat(80));
console.log(testMarkdown);
console.log("-".repeat(80));

// BROKEN: The OLD sanitization function that removed newlines
function sanitizeReportContent_BROKEN(content) {
  if (!content) return "";
  // THIS REMOVES NEWLINES! ASCII 10 (\n) is in range 0x00-0x1F
  let sanitized = content.replace(/[\x00-\x1F\x7F]/g, "");
  sanitized = sanitized
    .replace(/\r\n/g, "\n")
    .replace(/\u202E/g, "")
    .replace(/\u202D/g, "")
    .replace(/\u200B/g, "")
    .replace(/\u200C/g, "")
    .replace(/\u200D/g, "");
  return sanitized;
}

// FIXED: The corrected sanitization function that preserves newlines
function sanitizeReportContent_FIXED(content) {
  if (!content) return "";
  // Normalize line endings FIRST
  let sanitized = content.replace(/\r\n/g, "\n");
  // Remove ONLY problematic control characters, but PRESERVE newlines (ASCII 10) and tabs (ASCII 9)
  // Characters to remove: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, and 0x7F
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
  sanitized = sanitized
    .replace(/\u202E/g, "")
    .replace(/\u202D/g, "")
    .replace(/\u200B/g, "")
    .replace(/\u200C/g, "")
    .replace(/\u200D/g, "");
  return sanitized;
}

const broken = sanitizeReportContent_BROKEN(testMarkdown);
const fixed = sanitizeReportContent_FIXED(testMarkdown);

console.log("\n❌ BROKEN OUTPUT (all newlines removed):");
console.log("-".repeat(80));
console.log(broken);
console.log("-".repeat(80));
console.log("Length:", broken.length, "characters (was", testMarkdown.length, ")");

console.log("\n✅ FIXED OUTPUT (newlines preserved):");
console.log("-".repeat(80));
console.log(fixed);
console.log("-".repeat(80));
console.log("Length:", fixed.length, "characters (same as original", testMarkdown.length, ")");

console.log("\n📊 COMPARISON:");
console.log("-".repeat(80));
console.log("Original has", (testMarkdown.match(/\n/g) || []).length, "newlines");
console.log("Broken version has", (broken.match(/\n/g) || []).length, "newlines");
console.log("Fixed version has", (fixed.match(/\n/g) || []).length, "newlines");
console.log("\nOriginal === Fixed?", testMarkdown === fixed ? "✅ YES" : "❌ NO");
console.log("Broken === Original?", broken === testMarkdown ? "✅ YES" : "❌ NO");

console.log("\n" + "=".repeat(80));
console.log("CONCLUSION: Fixed version perfectly preserves the markdown structure!");
console.log("=".repeat(80));
