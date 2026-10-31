namespace MLoop.Tests;

/// <summary>A one-sheet Excel workbook built by hand (strings inline): the point is a data format that is bytes, not lines of text.</summary>
internal static class WorkbookFile
{
    public static string Write(string path, string[] header, IEnumerable<double[]> rows)
        => Write(path, header, rows.Select(r => r.Select(v => (object)v).ToArray()));

    /// <summary>The workbook holding the same table as a plain comma-separated file: a cell that is a number is a number, any other is text.</summary>
    public static string FromCsv(string csvPath, string workbookPath)
    {
        var lines = File.ReadAllLines(csvPath).Where(l => l.Length > 0).ToArray();
        var header = lines[0].Split(',');
        return Write(workbookPath, header, lines.Skip(1).Select(l => l.Split(',')
            .Select(cell => double.TryParse(cell, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var number) ? (object)number : cell)
            .ToArray()));
    }

    /// <summary>Cells are numbers (<see cref="double"/>) or text (anything else).</summary>
    public static string Write(string path, string[] header, IEnumerable<object[]> rows)
    {
        using var zip = System.IO.Compression.ZipFile.Open(path, System.IO.Compression.ZipArchiveMode.Create);
        void Entry(string entry, string xml)
        {
            using var writer = new StreamWriter(zip.CreateEntry(entry).Open(), new System.Text.UTF8Encoding(false));
            writer.Write(xml);
        }
        const string Main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
        const string Rel = "http://schemas.openxmlformats.org/package/2006/relationships";
        Entry("[Content_Types].xml", "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
            + "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            + "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>"
            + "<Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/></Types>");
        Entry("_rels/.rels", $"<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"{Rel}\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/></Relationships>");
        Entry("xl/workbook.xml", $"<?xml version=\"1.0\" encoding=\"UTF-8\"?><workbook xmlns=\"{Main}\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><sheets><sheet name=\"Sheet1\" sheetId=\"1\" r:id=\"rId1\"/></sheets></workbook>");
        Entry("xl/_rels/workbook.xml.rels", $"<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"{Rel}\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/></Relationships>");

        var sheet = new System.Text.StringBuilder($"<?xml version=\"1.0\" encoding=\"UTF-8\"?><worksheet xmlns=\"{Main}\"><sheetData><row r=\"1\">");
        for (var c = 0; c < header.Length; c++)
            sheet.Append($"<c r=\"{(char)('A' + c)}1\" t=\"inlineStr\"><is><t>{header[c]}</t></is></c>");
        sheet.Append("</row>");
        var rowNumber = 1;
        foreach (var row in rows)
        {
            rowNumber++;
            sheet.Append($"<row r=\"{rowNumber}\">");
            for (var c = 0; c < row.Length; c++)
                sheet.Append(row[c] is double number
                    ? System.FormattableString.Invariant($"<c r=\"{(char)('A' + c)}{rowNumber}\"><v>{number:R}</v></c>")
                    : $"<c r=\"{(char)('A' + c)}{rowNumber}\" t=\"inlineStr\"><is><t>{System.Security.SecurityElement.Escape(row[c]?.ToString())}</t></is></c>");
            sheet.Append("</row>");
        }
        sheet.Append("</sheetData></worksheet>");
        Entry("xl/worksheets/sheet1.xml", sheet.ToString());
        return path;
    }
}
