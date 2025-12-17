function executeScriptDba() {
Logger.log('Starting predictive process.');
try {
PredictiveMapDba.execute();
} catch (e) {
Logger.log(`Global Error: ${e.message}`);
}
}
const PredictiveMapDba = {
configDba: {
sheetName: 'Enrich',
sourceColumn: 'E',
destinationColumn: 'F',
startRow: 2,
map: {} // This will hold the dynamic predictive map driving decisions
},
execute: function () {
try {
const sheet =
SpreadsheetApp.getActiveSpreadsheet().getSheetByName(this.configDba.sheetName)
;
const dataRange = this.getRange(sheet, this.configDba.startRow);
this.buildPredictiveMap(dataRange);
this.processMap(sheet);
Logger.log("Predictive processing complete.");
} catch (e) {
Logger.log(`Execution Error: ${e.message}`);
}
},
getRange: function (sheet, startRow) {
const lastRow = sheet.getLastRow();

const sourceData = sheet.getRange(startRow,
this.columnLetterToIndex(this.configDba.sourceColumn), lastRow - startRow + 1,
1).getValues();
const destinationData = sheet.getRange(startRow,
this.columnLetterToIndex(this.configDba.destinationColumn), lastRow - startRow
+ 1, 1).getValues();
return { sourceData, destinationData };
},
buildPredictiveMap: function (dataRange) {
const { sourceData, destinationData } = dataRange;
const map = {};
for (let i = 0; i < sourceData.length; i++) {
const row = i + this.configDba.startRow;
const sourceValue = sourceData[i][0].trim();
const destinationValue = destinationData[i][0].trim();
map[row] = {
sourceValue,
destinationValue,
metadata: this.generateMetadata(sourceValue, destinationValue)
};
}
this.configDba.map = map; // Save the map for later use
},
generateMetadata: function (sourceValue, destinationValue) {
const isEmptySource = !sourceValue;
const isAttempted = sourceValue === "Attempted";
const isDestinationEmpty = !destinationValue;
return {
isFetchable: !isEmptySource && isDestinationEmpty && !isAttempted,
predictedOutcome: this.predictOutcome(sourceValue),
actionTag: this.determineAction(sourceValue, destinationValue)

};
},
predictOutcome: function (sourceValue) {
if (this.isValidUrl(sourceValue)) {
return "fetch";
} else if (sourceValue === "Attempted") {
return "skip";
}
return "mark_attempted";
},
determineAction: function (sourceValue, destinationValue) {
if (destinationValue === "" && sourceValue !== "Attempted") {
return "process";
} else if (sourceValue === "Attempted" && destinationValue === "") {
return "mark_attempted"; // Handle case where source is "Attempted" and
destination is blank
} else {
return "ignore";
}
},
processMap: function (sheet) {
const map = this.configDba.map;
const rowsToFetch = [];
const rowsToMarkAsAttempted = [];
// Collect rows that need fetching or marking
Object.keys(map).forEach(row => {
const dataPoint = map[row];
const { sourceValue, metadata } = dataPoint;
if (metadata.actionTag === "mark_attempted") {
rowsToMarkAsAttempted.push(row);
} else if (metadata.actionTag === "process" && metadata.isFetchable) {

rowsToFetch.push({ row, url: sourceValue });
}
});
// Process rows in parallel batches
this.fetchUrlsInParallel(sheet, rowsToFetch);
this.markRowsAsAttempted(sheet, rowsToMarkAsAttempted);
},
fetchUrlsInParallel: function (sheet, rowsToFetch) {
if (rowsToFetch.length === 0) return; // No URLs to fetch
// Prepare fetch requests in parallel
const requests = rowsToFetch.map(entry => ({
url: entry.url,
muteHttpExceptions: true
}));
const responses = UrlFetchApp.fetchAll(requests);
// Process responses and write to sheet in one batch
const updates = [];
responses.forEach((response, index) => {
const row = rowsToFetch[index].row;
const responseCode = response.getResponseCode();
let title;
if (responseCode >= 200 && responseCode < 300) {
const html = response.getContentText();
title = this.extractTitleFromHTML(html);
} else {
title = "Attempted";
}
updates.push({ row, value: title });
});

// Write all updates in one batch
this.writeBatch(sheet, updates);
},
writeBatch: function (sheet, updates) {
const destinationColumnIndex =
this.columnLetterToIndex(this.configDba.destinationColumn);
const ranges = updates.map(update => sheet.getRange(update.row,
destinationColumnIndex));
ranges.forEach((range, index) => {
range.setValue(updates[index].value);
});
},
markRowsAsAttempted: function (sheet, rows) {
const destinationColumnIndex =
this.columnLetterToIndex(this.configDba.destinationColumn);
const ranges = rows.map(row => sheet.getRange(row,
destinationColumnIndex));
ranges.forEach(range => {
range.setValue("Attempted");
});
},
extractTitleFromHTML: function (html) {
const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
let title = titleMatch && titleMatch[1] ? titleMatch[1].trim() : "No Title
Found";
return this.decodeHtmlEntities(title); // Decode HTML entities before
returning
},
decodeHtmlEntities: function (str) {

return str.replace(/&#(\d+);/g, (match, dec) => String.fromCharCode(dec))
// Convert numeric entities

.replace(/&([a-z]+);/gi, (match, entity) => {
const entities = {
amp: '&',
lt: '<',
gt: '>',
quot: '"',
apos: "'",
nbsp: ' ',
ndash: '–',
mdash: '—'
};
return entities[entity] || match;
}); // Convert named entities

},
columnLetterToIndex: function (letter) {
return letter.toUpperCase().charCodeAt(0) - 'A'.charCodeAt(0) + 1;
},
isValidUrl: function (url) {
const urlPattern = /^(https?:\/\/[^\s]+)/i;
return urlPattern.test(url);
}
};
