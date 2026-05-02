Descriptive Analysis:   
  
# Load required packages  
library(dplyr)  
library(tidyr)  
library(ggplot2)  
library(readr)  
  
# ==========================================  
# 1. DATA IMPORT & CLEANING  
# ==========================================  
# Load the specific CSV sheet you uploaded.   
# skip = 7 bypasses the Eurostat metadata at the top of the file.  
df <- read_csv("GDP per capita by NUTS 2 2007-2020.xlsx - Sheet 2.csv",   
               skip = 7, show_col_types = FALSE)  
  
# Clean the columns  
df_clean <- df %>%  
  # Keep the 'TIME' column (region names) and rename it to 'Region'  
  # Keep only columns that match a 4-digit year to drop the empty letter-flag columns  
  select(Region = TIME, matches("^20[0-9]{2}$")) %>%  
  # Remove the first row which is just the "GEO (Labels)" subtitle  
  slice(-1)  
  
# Reshape the data from Wide to Long format for analysis  
df_long <- df_clean %>%  
  pivot_longer(  
    cols = -Region,  
    names_to = "Year",  
    values_to = "GDP_Relative"  
  ) %>%  
  mutate(  
    Year = as.numeric(Year),  
    GDP_Relative = as.numeric(GDP_Relative),  
    # Define the Treatment variable based on the 75% cutoff rule  
    Treatment = ifelse(GDP_Relative < 75, 1, 0),  
    Treatment_Status = ifelse(Treatment == 1, "Treated (< 75%)", "Control (>= 75%)")  
  ) %>%  
  # Filter out rows with missing data and the large EU macro-aggregates  
  filter(!is.na(GDP_Relative),  
         !grepl("European Union", Region))  
  
# ==========================================  
# 2. SUMMARY STATISTICS (EVIDENCE OF VARIATION)  
# ==========================================  
# This table provides evidence that there are sufficient observations   
# in both the treatment and control groups to run an RDD.  
  
cat("--- Summary Statistics: Treatment vs. Control Regions (2007-2020) ---\n")  
summary_table <- df_long %>%  
  group_by(Treatment_Status) %>%  
  summarise(  
    `Total Observations` = n(),  
    `Mean GDP Relative` = round(mean(GDP_Relative), 2),  
    `Median GDP Relative` = median(GDP_Relative),  
    `Min GDP` = min(GDP_Relative),  
    `Max GDP` = max(GDP_Relative)  
  )  
  
print(summary_table)  
  
# ==========================================  
# 3. PRELIMINARY RDD DISTRIBUTION PLOT  
# ==========================================  
# Visualizing the distribution of the forcing variable (GDP) across regions  
  
rdd_forcing_plot <- ggplot(df_long, aes(x = GDP_Relative, fill = Treatment_Status)) +  
  geom_histogram(binwidth = 5, color = "black", alpha = 0.8) +  
  # Add the critical 75% RDD cutoff line  
  geom_vline(xintercept = 75, color = "red", linetype = "dashed", linewidth = 1.2) +  
  scale_fill_manual(  
    # Using colorblind-friendly colors  
    values = c("Treated (< 75%)" = "#E69F00", "Control (>= 75%)" = "#56B4E9"),  
    name = "Status"  
  ) +  
  labs(  
    title = "Distribution of Regional GDP per Capita (2007-2020)",  
    subtitle = "Evaluating variation around the 75% cutoff for EU Structural Funds Eligibility",  
    x = "Relative GDP per Capita (EU Average = 100%)",  
    y = "Number of Region-Year Observations",  
    caption = "Source: Eurostat | Red dashed line represents the RDD cutoff."  
  ) +  
  theme_minimal() +  
  theme(  
    plot.title = element_text(face = "bold", size = 14),  
    plot.subtitle = element_text(size = 11, color = "gray30"),  
    legend.position = "bottom",  
    axis.title = element_text(face = "bold")  
  )  
  
# Display the plot  
print(rdd_forcing_plot)  
  
# Save the high-quality image so you can embed it into your presentation slides!  
# ggsave("forcing_variable_distribution.png", plot = rdd_forcing_plot, width = 8, height = 5, dpi = 300)  
