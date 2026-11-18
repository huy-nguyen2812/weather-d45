/** Web App nhận dữ liệu từ Python và UPSERT theo cột "date" vào sheet daily_report.
 *  Cách dùng: Extensions → Apps Script → dán mã này → Project Settings → Script properties: TOKEN = <chuỗi bí mật dài>.
 *  Deploy → New deployment → Web app → Execute as: Me; Who has access: Anyone → copy URL (/exec). */
function doPost(e) {
  var out = function (o) { return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON); };
  try {
    var body = JSON.parse(e.postData.contents);
    var token = PropertiesService.getScriptProperties().getProperty('TOKEN');
    if (!token || body.token !== token) return out({ok: false, code: 401, error: 'unauthorized'});
    var rows = body.rows || [];
    if (!rows.length) return out({ok: false, code: 400, error: 'no rows'});
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sh = ss.getSheetByName('daily_report') || ss.insertSheet('daily_report');
    var cols = Object.keys(rows[0]);
    if (sh.getLastRow() === 0) sh.appendRow(cols);
    var keys = sh.getRange(2, 1, Math.max(sh.getLastRow() - 1, 1), 1).getValues().map(function (r) { return String(r[0]); });
    var inserted = 0, updated = 0;
    rows.forEach(function (r) {
      var line = cols.map(function (c) { return r[c] === null ? '' : r[c]; });
      var idx = keys.indexOf(String(r[cols[0]]));
      if (idx >= 0) { sh.getRange(idx + 2, 1, 1, cols.length).setValues([line]); updated++; }
      else { sh.appendRow(line); keys.push(String(r[cols[0]])); inserted++; }
    });
    return out({ok: true, code: 200, inserted: inserted, updated: updated});
  } catch (err) { return out({ok: false, code: 500, error: String(err)}); }
}
