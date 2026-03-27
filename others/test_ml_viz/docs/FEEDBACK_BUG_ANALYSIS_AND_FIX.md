# Feedback Button Tab Switch Bug - Root Cause Analysis & Fix

## Problem
When user submits feedback on the Data Viewer tab and then switches to another tab and back, the feedback button reappears asking for feedback again (instead of showing "Thank you" message).

## Root Cause Analysis

### Bug Chain:
1. **Index.tsx uses TabsContent** - Tabs unmount their content when switching away
2. **DataViewerContent unmounts** when user leaves Data Viewer tab
   - All React state is lost (including datasetHash)
   - localStorage still contains the feedback data
3. **User switches back to Data Viewer** - DataViewerContent remounts fresh
   - datasetHash state needs to be re-fetched from API
   - But during initial render, datasetHash is undefined
4. **FeedbackWidget renders before datasetHash is set**
   - storageKey = `datasetHash ? 'feedback_' + datasetHash : null` → null
   - useEffect checks `if (storageKey)` - condition is FALSE
   - localStorage lookup SKIPPED
   - Form shows instead of "Thank you" message
5. **Race condition**: API fetch completes, datasetHash updates, useEffect runs, localStorage check finally succeeds... but too late, user already sees the form

### Why previous fix didn't work:
- Previous fix checked `if (datasetHash)` but datasetHash was undefined during tab transition
- datasetHash depends on an API call from `/api/features/{runId}` which has network delay
- Component unmounting/remounting breaks this timing

## Solution

### Use runId instead of datasetHash as localStorage key

**Advantages:**
- ✅ `runId` comes from PipelineContext which persists across tab switches
- ✅ `runId` is immediately available (no API call needed)
- ✅ `runId` is guaranteed to be the same for a given dataset run
- ✅ Works correctly even if component unmounts/remounts

### Implementation:
```typescript
// FeedbackWidget.tsx
const { runId } = usePipeline(); // Get runId from context (always available!)

// Create persistent key using runId (survives tab switches)
const storageKey = runId ? `feedback_run_${runId}` : (datasetHash ? `feedback_${datasetHash}` : null);

// Use this key everywhere in localStorage
useEffect(() => {
  if (storageKey) {
    const stored = localStorage.getItem(storageKey);
    // ... restore feedback
  }
}, [storageKey]); // Depends on runId which never changes

const handleSubmit = async (label: number) => {
  // ... submit
  localStorage.setItem(storageKey, {...}); // Use consistent key
}
```

## Files Changed
- **frontend/src/components/FeedbackWidget.tsx**
  - Added `import { usePipeline } from "../context/PipelineContext"`
  - Changed storage key from `datasetHash`-based to `runId`-based
  - Updated useEffect dependency array to use `storageKey`
  - Updated handleSubmit to use new storage key

## Testing Scenario

```
1. Run pipeline on Data Viewer tab → Displays dataset with quality score
2. Submit feedback: "Perfect" → Shows "Thank you for your feedback!"
3. Switch to "Visual Gallery" tab  → Data Viewer unmounts
4. Switch back to "Data Viewer" tab → Data Viewer remounts
5. ✅ EXPECTED: Still shows "Thank you for your feedback!" (from localStorage)
6. ✅ ACTUAL: Should now show "Thank you" (was previously showing form again)
```

## Why This Is A Better Design

| Aspect | Old (datasetHash) | New (runId) |
|--------|-------------------|------------|
| Availability | Requires API fetch + network delay | Immediately from context |
| Persistence | Lost on tab switch | Always available |
| Deterministic | Depends on API | Always consistent |
| Component lifecycle | Vulnerable to unmounting | Unaffected |
| Race conditions | Yes (timing issues) | No |

## Summary
The feedback state now **survives tab switches** because it's keyed by `runId` (from context) which is always available, rather than `datasetHash` (from slow API) which becomes undefined when the component unmounts.
