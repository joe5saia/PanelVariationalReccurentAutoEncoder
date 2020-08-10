###################################################################################################
#=
This script reads in compustat data from wrds and makes a dataset to train our Neural Net on.
The output dataset contains ~ 5000 firms each with exactly 40 observations. To make this dataset 
we download all our firms and then filter out firms that do not have 40 consecutive observations.
If a firm is missing one or two observations for a single series then we fill it in with linear 
interpolation or by a firm level mean. The output data set is data_40.csv
=#
###################################################################################################
using DataFrames, CSV, LibPQ, Dates, Statistics, LinearAlgebra, Revise, StatsBase

includet("pulldata_helper.jl")

## Uses password in pgpass file
cstring = open("wrds_cstring.txt") do file
    read(file, String)
end
conn = LibPQ.Connection(cstring)

# Pull in the compustat data
compfilter = "fic = 'USA' AND CONSOL = 'C' AND popsrc = 'D' AND indfmt = 'INDL' AND datafmt = 'STD' AND curcdq = 'USD'"
data_sql = "select a.gvkey, a.datadate, b.sich, ppentq, atq, actq, intanq, cheq, dlcq, dlttq, ltq,
cogsq, xoprq, revtq, saleq, nopiq, txditcq, aqcy, ceqq, seqq, dvpq, prccq, cshoq
FROM comp.fundq  AS a 
LEFT JOIN ( SELECT gvkey, datadate, floor(sich/100) as sich FROM comp.co_industry WHERE consol = 'C' AND popsrc='D') AS b
ON a.gvkey = b.gvkey AND a.datadate = b.datadate
WHERE
a.gvkey in (SELECT DISTINCT gvkey
    FROM comp.fundq 
    WHERE $(compfilter)
    AND datadate >= '1990-01-01'
    GROUP BY gvkey
    HAVING count(atq) > 39)
AND $(compfilter)
AND datafqtr IS NOT NULL
ORDER BY gvkey, datadate
;";

df = DataFrame(execute(conn, data_sql))
## Make copies of the dataframe to avoid redownloading the data during debugging
dfzzz = copy(df)
df = copy(dfzzz)
###############################################################################
# Create Quarterly data and deal with duplicates from fiscal year changes
###############################################################################
df[!, :datadateq] = normdateQ.(df[!, :datadate])

# Find last datadate for firms with dupe obs within quarter
gdf = groupby(df, [:gvkey, :datadateq])
datadf = combine( groupby(df, [:gvkey, :datadateq]), :datadate => last => :datadate)

# Aggregate values for firms with dupe obs within quarter
# Take aveverage of available observations
gdf = groupby(select(df, Not(:datadate)), [:gvkey, :datadateq])
df = combine(gdf,  valuecols(gdf) .=> mean .=> valuecols(gdf))

# Merge back on datadate
df = leftjoin(df, datadf, on=[:gvkey, :datadateq])

# Make lead datadate
df[!, :F_datadateq] = df[!, :datadateq] .+ Dates.Month(3)

###############################################################################
## Assign SIC code to every date and make sector dummies
###############################################################################
for subdf in groupby(df, :gvkey)
    # Save sic variable as first valid sich observation 
    # If sich is missing for an observation then set it to sic
    # If sich is no missing then update sic 
    # If no valid sich then set to -1
    indexs = findall(.!ismissing, subdf[:,:sich])
    >(length(indexs), 0) ? sic = subdf[first(indexs), :sich] : sic = -1.0
    for i in axes(subdf, 1)
        if ismissing(subdf[i, :sich])
            subdf[i, :sich] = sic
        else
            sic =  subdf[i, :sich]
        end
    end
end


# Sector Dummies
df[!, :sector] .= 0
for i in 1:size(df, 1)
    sic = df[i, :sich]
    if 0 <= sic < 10
        df[i, :sector] = 1
    elseif 10 <= sic <= 14
        df[i, :sector] = 2
    elseif 15 <= sic <= 17
        df[i, :sector] = 3
    elseif 20 <= sic <= 39
        df[i, :sector] = 4
    elseif 40 <= sic <= 49
        df[i, :sector] = 5
    elseif 50 <= sic <= 51
        df[i, :sector] = 6
    elseif 52 <= sic <= 59
        df[i, :sector] = 7
    elseif 70 <= sic <= 89
        df[i, :sector] = 8
    elseif 60 <= sic <= 67
        df[i, :sector] = -2
    elseif 91 <= sic <= 97
        df[i, :sector] = -1
    end
end




###########################################################################################
## Fill in missings using a few techniques.
## keep iterating till we can not logically fill in any more data
###########################################################################################
global nmissqtq = 0
global counts = 0
while nmissqtq !== sum(ismissing.(df.atq))
    global nmissqtq = sum(ismissing.(df.atq))
    @show global counts += 1
    ###########################################################################################
    ## Use accounting identities to fill missings with factor structure
    ###########################################################################################
    # Some variables we can back out by accounting identities that imply factor structure
    # Assests = liabilities + equity
    # Two measures of equity, both different measures of "equity" that should be mostly interchangable
    for subdf in groupby(df, :gvkey) 
        factorfill!(subdf[!, :seqq], subdf[!, :ceqq])
        factorfill!(subdf[!, :atq], subdf[!, :ltq], subdf[!, :seqq])
    end

    ###########################################################################################
    ## Linearly interpolate quarterly data where it is stated only annually
    ###########################################################################################
    for subdf in groupby(df, :gvkey)
        for varname in Symbol.(names(df))
            interpolate!(subdf[!, varname])
        end
    end
end

###########################################################################################
## Conditional on other data present, we can assume the following are 0 (We'll filter on atq later)
###########################################################################################
df[ismissing.(df[!, :aqcy]), :aqcy] .= 0
df[ismissing.(df[!, :intanq]), :intanq] .= 0
df[ismissing.(df[!, :actq]), :intanq] .= 0
df[ismissing.(df[!, :saleq]), :intanq] .= 0
df[ismissing.(df[!, :dlcq]), :intanq] .= 0
df[ismissing.(df[!, :dlttq]), :intanq] .= 0

###########################################################################################
## Make some variables
###########################################################################################
# Investment
df[!, :investq] = missings(Float64, size(df, 1))
for subdf in groupby(df, :gvkey)
    for j in 4:size(subdf,1)
        subdf[j-3, :investq] = (subdf[j, :ppentq] - subdf[j-3, :ppentq]) / max(subdf[j-3, :ppentq], 0.0001)
    end
end

# Max all observations have at least a small but positive level of assets
df[!, :atq] = max.(df[!, :atq], 0.0001)
# Size
df[!, :sizeq] = log.(df[!, :atq])
# Total Debt
df[!, :dtq] = df[!,:dlcq] + df[!,:dlttq]
## Asset Leverage
df[!, :alevq] = df[!, :dtq] ./ df[!,:atq]
## Net Leverage
df[!, :nlevq] = (df[!,:dlcq] + df[!,:dlttq] - df[!,:actq]) ./ df[:,:atq]
## Cash Liquidity
df[!, :cliqq] = df[!,:cheq]./ df[!, :atq]
## Current Asset Liquidity
df[!, :aliqq] = df[!, :actq]./ df[!,:atq]
## Capital ratio
df[!, :caprq] = df[!, :seqq] ./ df[!, :atq]
# Dividend Payer
df[!,:divpayq] = Int.(.!ismissing.(df[!,:dvpq]) .& (df[!,:dvpq] .> 0))
# Tobin's q
df[!, :mceqq] = df[!, :prccq] .* df[!,:cshoq]
df[!, :batq] = df[!, :atq] + df[!,:ceqq] - df[!, :mceqq] + df[!, :txditcq]
df[!, :tqq] = df[!, :atq] ./ df[!, :batq]
# Current asset ratio
df[!, :carq] = df[!,:actq] ./ df[!, :atq]
# Sales ratio
df[!, :salerq] = df[!, :saleq] ./ df[!, :atq]
# revenue ratio
df[!, :revtrq] = df[!, :revtq] ./ df[!, :atq]
# operating expense ratio
df[!, :xoprrq] = df[!, :xoprq] ./ df[!, :atq]
# Aquistion ratio
df[!, :aqcyrq] = df[!, :aqcy] ./ df[!, :atq]
# nonoperating ratio
df[!, :nopirq] = df[!, :nopiq] ./ df[!, :atq]
# Intangabile ratio
df[!, :intanrq] = df[!, :intanq] ./ df[!, :atq]

###########################################################################################
## Drop observations that we don't want to use
###########################################################################################
# Keep data after 1990
df = sort(filter(row -> row.datadateq .> Date(1990), df), [:sich, :gvkey, :datadateq])
# Keep only non financials
filter!(row -> row.sector > 0 , df)

###########################################################################################
## Drop data where we have no way to find assets
###########################################################################################
# If assests are 100% missing then there's no hope
df[!, :allmissing] .= false
for subdf in groupby(df, :gvkey) 
    subdf[:, :allmissing] .= all(ismissing.(subdf[:, :atq]))
end
filter(row -> !row.allmissing, df)
# Data can be sparse at the begining and end of firm's time in compustat. Throw out these end points based on when assets start and end
df[!, :started] .= false
for subdf in groupby(df, :gvkey) 
    started = false
    for i in 1:size(subdf, 1) 
        started |= !ismissing(subdf[i, :atq])
        subdf[i, :started] = started
    end
end
df[!, :ended] .= true
for subdf in groupby(df, :gvkey) 
    ended = true
    for i in reverse(1:size(subdf, 1) )
        ended &= ismissing(subdf[i, :atq])
        subdf[i, :ended] = ended
    end
end
filter!(row -> row.started & !row.ended, df)
select!(df, Not([:started, :ended, :allmissing]))

###########################################################################################
# Split companies into a new company when they go dark for more than a year
###########################################################################################
for subdf in groupby(df, :gvkey)
    suffix = 'A'
    subdf[1, :gvkey] *= suffix
    for i in 2:size(subdf,1)
        if xor(ismissing(subdf[i,:atq]), ismissing(subdf[i-1,:atq])) 
            suffix += 1
        end
        subdf[i, :gvkey] *= suffix
    end
end

###########################################################################################
## Drop data where we have no way to find assets for the split companies
###########################################################################################
df[!, :allmissing] .= false
for subdf in groupby(df, :gvkey) 
    subdf[:, :allmissing] .= all(ismissing.(subdf[:, :atq]))
end
filter(row -> !row.allmissing, df)
# Data can be sparse at the begining and end of firm's time in compustat. Throw out these end points based on when assets start and end
df[!, :started] .= false
for subdf in groupby(df, :gvkey) 
    started = false
    for i in 1:size(subdf, 1) 
        started |= !ismissing(subdf[i, :atq])
        subdf[i, :started] = started
    end
end
df[!, :ended] .= true
for subdf in groupby(df, :gvkey) 
    ended = true
    for i in reverse(1:size(subdf, 1) )
        ended &= ismissing(subdf[i, :atq])
        subdf[i, :ended] = ended
    end
end
filter!(row -> row.started & !row.ended, df)
select!(df, Not([:started, :ended, :allmissing]))


###########################################################################################
## Keep comapnies where we have at least 3 years of data
# use an inner join for speed. 
###########################################################################################
df = innerjoin(df, DataFrame(gvkey = filter(row -> row.atq_length >= 40, combine(groupby(df, :gvkey), :atq => length))[!, :gvkey]), on = :gvkey)

# Select the variables we want
select!(df, [:gvkey, :datadateq, :sector, :sizeq, :divpayq, :investq, :alevq, :nlevq, :carq, :salerq, :revtrq, :xoprrq, :aqcyrq, :nopirq, :intanrq])

# Truncate the ratios to have a maximum value of 1 and minimum value of -2
for varname in [:alevq, :nlevq, :carq, :salerq, :revtrq, :xoprrq, :aqcyrq, :nopirq, :intanrq]
    df[.!ismissing.(df[!, varname]) .& (df[!, varname] .> 1), varname ] .= 1
    df[.!ismissing.(df[!, varname]) .& (df[!, varname] .< -2), varname ] .= -2
end

# Make size var 1
df[!, :sizeq] ./= std(skipmissing(df[!, :sizeq]))

# truncated investment at around 2.5 
p = Statistics.quantile(skipmissing(df[!,:investq]), 0.975)
df[.!ismissing.(df.investq) .& (df.investq .> p), :investq] .= p
df[.!ismissing.(df.investq) .& (df.investq .< -1), :investq] .= -1

###########################################################################################
## Drop data at the ends where we don't have investment data by design
###########################################################################################
df[!, :allmissing] .= false
for subdf in groupby(df, :gvkey) 
    subdf[:, :allmissing] .= all(ismissing.(subdf[:, :investq]))
end
filter(row -> !row.allmissing, df)
# Data can be sparse at the begining and end of firm's time in compustat. Throw out these end points based on when assets start and end
df[!, :started] .= false
for subdf in groupby(df, :gvkey) 
    started = false
    for i in 1:size(subdf, 1) 
        started |= !ismissing(subdf[i, :investq])
        subdf[i, :started] = started
    end
end
df[!, :ended] .= true
for subdf in groupby(df, :gvkey) 
    ended = true
    for i in reverse(1:size(subdf, 1) )
        ended &= ismissing(subdf[i, :investq])
        subdf[i, :ended] = ended
    end
end
filter!(row -> row.started & !row.ended, df)
select!(df, Not([:started, :ended, :allmissing]))



# Fill in remaining missings with median value from each firm. Introduces NaN when all observations are missing for a firm
for varname in [:sizeq, :divpayq, :investq, :alevq, :nlevq, :carq, :salerq, :revtrq, :xoprrq, :aqcyrq, :nopirq, :intanrq]
    ms = combine(groupby(df, :gvkey), varname => mean ∘ skipmissing => :m)
    for i in 1:size(df,1)
        if ismissing(df[i, varname])
            #println( ms)
            df[i, varname] = ms[ms.gvkey .== df[i, :gvkey], :m][1]
        end
    end
end

# Hack to NaN's to an out of range number
for varname in [:sizeq, :divpayq, :investq, :alevq, :nlevq, :carq, :salerq, :revtrq, :xoprrq, :aqcyrq, :nopirq, :intanrq]
    df[isnan.(df[!, varname]), varname] .= -99999
end

# Replace the remaining missing values with the median value from their sector
for varname in [:sizeq, :divpayq, :investq, :alevq, :nlevq, :carq, :salerq, :revtrq, :xoprrq, :aqcyrq, :nopirq, :intanrq]
    ms = combine(groupby(df, :sector), varname => median ∘ skipmissing => :m)
    for i in 1:size(df,1)
        if df[i, varname] == - 99999
            df[i, varname] = ms[ms.sector .== df[i, :sector], :m][1]
        end
    end
end

# Keep only the first 40 observations
df = innerjoin(df, DataFrame(gvkey = filter(row -> row.sizeq_length >= 40, combine(groupby(df, :gvkey), :sizeq => length))[!, :gvkey]), on = :gvkey)

df[!, :rank] .= 1
for subdf in groupby(df, :gvkey)
    subdf[:, :rank] .= denserank(subdf[:, :datadateq])
end

filter!(row -> row.rank <= 40, df)
sort!(df, [:rank, :gvkey])

# Save Data
CSV.write("data_40.csv", df)

