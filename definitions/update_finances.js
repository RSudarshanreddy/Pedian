// publish("tbl_daily_fin_updates", {
//     type: "incremental",
//     schema: "dataform",
//     tags: ["Forecast"]
// }).query(`
//   SELECT 
//     Stock, 
//     cmp, 
//     previous_change, 
//     _1_day_change
//   FROM 
//     \`sudarshan-442212.fin.finances\`
// `);

operate("update_finances_from_gsheets")
    .tags("Forecast")
    .queries(` MERGE INTO \`sudarshan-442212.dataform.tbl_daily_fin_updates\` AS Target
      USING (
        SELECT 
          Stock,
          cmp,
          previous_changes,
          _1_day_changes 
        FROM \`sudarshan-442212.fin.finances\`
      ) AS Source
      ON Source.Stock = Target.Stock
      WHEN MATCHED THEN
        UPDATE SET 
          Target.cmp = Source.cmp,
          Target.previous_change = Source.previous_changes,
          Target._1_day_change = Source._1_day_changes
      WHEN NOT MATCHED THEN
        INSERT (
          Stock,
          cmp,
          previous_change,
          _1_day_change
        )
        VALUES (
          Source.Stock,
          Source.cmp,
          Source.previous_changes,
          Source._1_day_changes              
              )
        `);
