operate("update_finances_from_gsheets")
    .tags("Forecast")
    .queries(`MERGE INTO \`sudarshan-442212.dataform.tbl_daily_fin_updates\` AS Target
      USING (
        SELECT            
            Stock,
            Buy_Qty,
            Buy_Price,
            cmp, 
            Target,
            Buy_Value, 
            Current_Market_Value,
            Sell_Target_value,
            previous_changes, 
            _1_day_changes,
            Highest_Cday, 
            Lowest_Cday,
            _52_high,
            _52_low

        FROM \`sudarshan-442212.fin.Finances\`
      ) AS Source

      ON Source.Stock = Target.Stock

      WHEN MATCHED THEN
        UPDATE SET 
          Target.cmp = Source.cmp,
          Target.previous_changes = Source.previous_changes,
          Target._1_day_change = Source._1_day_changes,
          Target.Highest_Cday = source.Highest_Cday,
          Target.Lowest_Cday = source.Lowest_Cday,
          Target._52_high = source._52_high,
          Target._52_low = source._52_low,
           #Target.Buy_Qty = source.Buy_Qty,
           Target.Buy_Value = source.Buy_Value,
           #Target.Buy_Price = source.Buy_Price,          
           target.Current_Market_Value = source.Current_Market_Value ,
           target.Sell_Target_value = source.Sell_Target_value,       
          target.calday = CURRENT_DATE()
      WHEN NOT MATCHED THEN
        INSERT (
           Stock,
           # Buy_Qty,
           #Buy_Price,
           cmp, 
           # Target,
           Buy_Value, 
           Current_Market_Value,
           Sell_Target_value,
           previous_changes, 
           _1_day_change,
           Highest_Cday, 
           Lowest_Cday,
           _52_high,
           _52_low, 
           calday
        )
        VALUES  (
            source.Stock,
            #source.Buy_Qty,
            #source.Buy_Price,
            source.cmp, 
            #source.Target,
            source.Buy_Value, 
            source.Current_Market_Value,
            source.Sell_Target_value,
            source.previous_changes, 
            source._1_day_changes,
            source.Highest_Cday, 
            source.Lowest_Cday,
            source._52_high,
            source._52_low, 
            CURRENT_DATE()        
          
          )            
              `)
    .queries(
        `UPDATE \`sudarshan-442212.dataform.tbl_daily_fin_updates\` AS C
              SET C.Buy_Value = B.Latest_Buy_Value -- , C.Buy_Qty = B.Latest_Buy_Qty
              FROM (
                    SELECT A.Stock, B.Latest_Buy_Value, B.Entity
                    FROM \`sudarshan-442212.fin.Finances\` AS A
                    LEFT JOIN \`sudarshan-442212.fin.fin_squares\` AS B
                    ON A.Stock = B.Entity
                    WHERE B.Entity IS NOT NULL
              ) AS B
              WHERE C.Stock = B.Stock;
              `);
