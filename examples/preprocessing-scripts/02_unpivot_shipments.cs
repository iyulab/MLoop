using MLoop.Extensibility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// Example: Wide-to-Long transformation (Unpivot)
/// Demonstrates: Converting multiple date/quantity columns into rows
/// Use case: Dataset 006 (표면처리 공급망최적화) with 1차~10차 출고날짜/출고량
/// </summary>
public class UnpivotShipments : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        ctx.Logger.Info("=== Wide-to-Long Transformation (Unpivot) ===");

        // Read input data
        var data = await ctx.Csv.ReadAsync(ctx.InputPath);
        ctx.Logger.Info($"Loaded: {data.Count:N0} rows (wide format)");

        // Unpivot: Convert multiple columns (1차~10차) into rows
        var unpivoted = new List<Dictionary<string, string>>();
        int totalShipments = 0;

        foreach (var row in data)
        {
            // Base columns that remain the same
            var baseData = new Dictionary<string, string>
            {
                ["생산일자"] = row.ContainsKey("생산일자") ? row["생산일자"] : "",
                ["작업지시번호"] = row.ContainsKey("작업지시번호") ? row["작업지시번호"] : "",
                ["제품코드"] = row.ContainsKey("제품코드") ? row["제품코드"] : "",
                ["시작"] = row.ContainsKey("시작") ? row["시작"] : "",
                ["종료"] = row.ContainsKey("종료") ? row["종료"] : "",
                ["생산량(Kg)"] = row.ContainsKey("생산량(Kg)") ? row["생산량(Kg)"] : ""
            };

            // Unpivot 10 shipment pairs (date + quantity)
            for (int i = 1; i <= 10; i++)
            {
                var dateKey = $"{i}차 출고날짜";
                var qtyKey = $"{i}차 출고량";

                if (row.ContainsKey(dateKey) && !string.IsNullOrWhiteSpace(row[dateKey]))
                {
                    var shipmentRow = new Dictionary<string, string>(baseData)
                    {
                        ["출고순번"] = i.ToString(),
                        ["출고날짜"] = row[dateKey],
                        ["출고량"] = row.ContainsKey(qtyKey) ? row[qtyKey] : "0"
                    };

                    unpivoted.Add(shipmentRow);
                    totalShipments++;
                }
            }
        }

        ctx.Logger.Info($"Unpivoted: {data.Count:N0} wide rows → {unpivoted.Count:N0} long rows");
        ctx.Logger.Info($"Total shipments extracted: {totalShipments:N0}");

        // Save output
        var outputPath = Path.Combine(ctx.OutputDirectory, "02_unpivoted_shipments.csv");
        await ctx.Csv.WriteAsync(outputPath, unpivoted);

        ctx.Logger.Info($"✅ Saved: {outputPath}");
        return outputPath;
    }
}
