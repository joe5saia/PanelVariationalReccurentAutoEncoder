###################################################################################################
#=
Functions used in pulldata.jl
=#
###################################################################################################
function keyarray(x)
    out = "(\'" * string(x[1]) * "\'"
    for i in x[2:end]
        out *= ", \'" * string(i) * "\'"
    end
    out *= ")"
    return out
  end
  
  function joelog(x)
    if ismissing(x)
        return missing
    elseif x > 0
        return log(x)
    else
        return missing
    end
  end
  
  
  function ltmiss(x,y)
    # test if x < y. If missing return false
    ismissing(x) && return false
    ismissing(y) && return false
    (x <= y ) && return true
    return false
  end
  
  ###############################################################################
  # Date Functions
  ###############################################################################
  
  function normdateQ(d)
    # Make dates align to first day of third month of quarter
    y = year(d)
    q = quarterofyear(d)
    return Date(y, 3*q)
  end
  
  function normdateM(date)
    # Make dates align to first day of month
    Date(year(date), month(date))
  end
  
  # Winsorizing
  function windsorize!(x, p)
    threshhold = quantile(skipmissing(x), p)
    x[ltmiss(x, threshhold)] .= threshhold 
    threshhold = quantile(skipmissing(x), 1 - p)
    x[ltmiss(threshhold, x)] .= threshhold 
  end
  
  ###############################################################################
  # Aggregator functions
  ###############################################################################
  function compaggmean(x)
    if all(ismissing.(x))
      return missing
    else
      return mean(skipmissing(x))
    end
  end
  
  function probit2(x)
    if ismissing(x)
        return x
    else
        return cdf(Normal(), x)
    end
  end

  function factorfill!(x, y)
    xm = ismissing.(x)
    ym = ismissing.(y)
    xyznm = .!(xm .| ym)
    if sum(xyznm) >= 1 
        xmean = mean(x[xyznm])
        ymean = mean(y[xyznm])
        x[:] .-= xmean
        y[:] .-= ymean
        x[xm] .= y[xm] * reg(y[xyznm],  x[xyznm]) 
        y[ym] .= x[ym] * reg(y[xyznm],  x[xyznm]) 
        x[:] .+= xmean 
        y[:] .+= ymean 
    end
  end

function factorfill!(x, y, z)
    xm = ismissing.(x)
    ym = ismissing.(y)
    zm = ismissing.(z)
    xyznm = .!(xm .| ym .| zm)
    if sum(xyznm) >= 2 
        xmean = mean(x[xyznm])
        ymean = mean(y[xyznm])
        zmean = mean(z[xyznm])
        x[:] .-= xmean
        y[:] .-= ymean
        z[:] .-= zmean
        x[xm] .= [y[xm] z[xm]] * reg([y[xyznm] z[xyznm]],  x[xyznm]) 
        y[ym] .= [x[ym] z[ym]] * reg([y[xyznm] z[xyznm]],  x[xyznm]) 
        z[zm] .= [x[zm] y[zm]] * reg([y[xyznm] z[xyznm]],  x[xyznm]) 
        x[:] .+= xmean 
        y[:] .+= ymean 
        z[:] .+= zmean 
    end
end

function reg(x, y)
    return ((x' * x) + 0.000001 *I)\x' * y
end

function interpolate!(x, maxmiss = 4)
  nmiss = 0
  lagged = x[1]
  for i in Iterators.rest(eachindex(x),1)
    if !ismissing(x[i])
      if nmiss <= maxmiss
        # Don't interpolate if more than maxmiss quarters of data is missing
        for (j,k) in enumerate(i-nmiss:i-1)
          x[k] = lagged + j * (x[i] - lagged)/(nmiss+1)
        end
      end
      lagged = x[i]
      nmiss = 0
    else
      nmiss += 1
    end
  end
end
