// Shared mutable UI state.
//
// These values used to be globals shared by the classic scripts; now that
// each file is an ES module, the state they share lives in one exported
// object. Reads and writes go through `state.x`, so a call site makes it
// obvious when a value outlives the current module.
export const state = {
  activeId: null as string | null,
  appInitialized: false as boolean,
  currentMeeting: null as any,
  editing: null as string | null,
  meetings: [] as any[],
  modelsReady: false as boolean,
  pendingMeeting: null as any,
  searchQuery: "" as string,
  setupReadyResolve: null as (() => void) | null,
  uploading: false as boolean,
  viewingVersion: null as number | null,
};
