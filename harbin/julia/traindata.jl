using HDF5, JSON, JLD2, FileIO, Dates, Printf
include("traffic.jl")
include("Trip.jl")
include("SpatialRegion.jl")
include("Link.jl")

function collectslotdata(links::Dict, region::SpatialRegion, trips::Vector{Trip}, tms::T) where T
    """
    Collect the trips in the past 30 minutes of `tms` to create the traffic tensor
    along with the trips in the future 40 minutes.
    """
    slotsubtrips, subtrip_indices = timeslotsubtrips(trips, tms-30*60, tms)
    # generate the traffic heap map data
    createflowtensor!(region, slotsubtrips)
    # generate the traffic flow data per link
    max_val = createflowtensorlink!(links, slotsubtrips, subtrip_indices)
    slottrips = timeslottrips(trips, tms, tms+46*60)
    ##
    slottrips = filter(t->t.tms[end]-t.tms[1]>=7*60, slottrips)
    copy(region.S), copy(region.I), copy(region.O), slottrips, max_val
end

function savetraindata(h5file::String, links::Dict, region::SpatialRegion, trips::Vector{Trip}, stms::T) where T
    """
    trips: all trips in the day
    stms: start unix time of the day
    """
    for i = 1:length(trips)
        # remove redundant points
        non_red = removeredundantpoints(trips[i])
        trips[i].tms = non_red.tms
        trips[i].lon = non_red.lon
        trips[i].lat = non_red.lat
    end
    println("finished removing redundant points")
    println("Total number of trips: " * string(length(trips)))
    
    trips = filter(isvalidtrip_fmm, trips)
    println("finished removing invalid fmm trips")
    println("Total number of trips after removal: " * string(length(trips)))
    
    range = 30*60:20*60:24*3600 # maybe change 20 mins to 10 mins
    # iterate through all slots to get the maximum value for I, O and S
    max_norm = [0.0, 0.0, 0.0]
    for (slot, tms) in enumerate(range)
        S, I, O, slottrips, max_val = collectslotdata(Dict([]), region, trips, stms+tms)
        for i = 1:3
            if max_val[i] > max_norm[i]
                max_norm[i] = max_val[i]
            end
        end  
    end
    println("Maximum values: " * string(max_norm[1]) * " " * string(max_norm[2]) * " " *  string(max_norm[3]))
        
    hasratio = isdefined(trips[1], :endpointsratio)
    h5open(h5file, "w") do f
        range = 30*60:20*60:24*3600 # maybe change 20 mins to 10 mins
        for (slot, tms) in enumerate(range)
            S, I, O, slottrips, _ = collectslotdata(links, region, trips, stms+tms)
            f["/$slot/S"] = S
            f["/$slot/I"] = I
            f["/$slot/O"] = O
#             # code for normalizing the data by the max value at that day
#             for (gid, link) in links
#                 link.I /= max_norm[1]
#                 link.O /= max_norm[2]
#                 link.S /= max_norm[3]
#             end
            f["/$slot/Links"] = JSON.json(links)
            f["/$slot/ntrips"] = length(slottrips)
            for i = 1:length(slottrips)
#                 # if we don't perform the removeredundantpoints operation the dimension of tms and indices won't match
#                 if size(slottrips[i].tms, 1) != size(slottrips[i].indices, 1) || size(slottrips[i].tms, 1) != size(slottrips[i].opath, 1)
#                     println(size(slottrips[i].tms, 1))
#                     println(size(slottrips[i].indices, 1))
#                     println(size(slottrips[i].opath, 1))
#                 end
                f["/$slot/trip/$i"] = convert(Array{Int32}, slottrips[i].roads)
                f["/$slot/time/$i"] = (slottrips[i].tms[end]-slottrips[i].tms[1])/60.0
                ## origin and destination
                f["/$slot/orig/$i"] = [slottrips[i].lon[1]-region.minlon, slottrips[i].lat[1]-region.minlat]
                f["/$slot/dest/$i"] = [slottrips[i].lon[end]-region.minlon, slottrips[i].lat[end]-region.minlat]
                f["/$slot/lon/$i"] = slottrips[i].lon .- region.minlon
                f["/$slot/lat/$i"] = slottrips[i].lat .- region.minlat
                f["/$slot/ratio/$i"] = hasratio ? slottrips[i].endpointsratio : [1.0, 1.0]
                ## path distance
                f["/$slot/distance/$i"] = pathdistance(slottrips[i])
                # store data at the mid-point of each trip
                f["/$slot/times/$i"] = slottrips[i].tms ./ 60.0
                f["/$slot/distances/$i"] = pathdistanceall(slottrips[i])
                f["/$slot/opath/$i"] = convert(Array{Int32}, slottrips[i].opath)
                f["/$slot/mgeom_wkt/$i"] = slottrips[i].mgeom_wkt
                f["/$slot/pgeom_wkt/$i"] = slottrips[i].pgeom_wkt
                f["/$slot/indices/$i"] = convert(Array{Int32}, slottrips[i].indices)
                f["/$slot/offset/$i"] = convert(Array{Float64}, slottrips[i].offset)
                f["/$slot/spdist/$i"] = convert(Array{Float64}, slottrips[i].spdist)
            end
# #             println("number of trips: " * string(length(slottrips)))
#             println("number of links with feature: " * string(length(collect(keys(links)))))
            links = Dict([])
        end
    end
end

function savetraindata(h5path::String, links::Dict, region::SpatialRegion, jldfile::String)
    """
    Dump the trips in `jldfile` into train data.
    """
    ymd = basename(jldfile) |> splitext |> first |> x->split(x, "_") |> last
    # chengdu dataset uses 4 numbers for year, while harbin uses 2
#     m, d = parse(Int, ymd[3:4]), parse(Int, ymd[5:6])
    m, d = parse(Int, ymd[5:6]), parse(Int, ymd[7:8])
    h5file = ymd * ".h5"
    ## Filtering out the trajectories with serious GPS errors
    @printf("%s\n", jldfile)
    trips = filter(trip -> length(trip.roads)>=5, load(jldfile, "trips"))
    # chengdu dataset is in 2016, while harbin is in 2015
    stms = Dates.datetime2unix(DateTime(2016, m, d, 0, 0))
    savetraindata(joinpath(h5path, h5file), links, region, trips, stms)
end

function savetraindata(h5path::String, jldpath::String)
    """
    Dump all jldfiles into h5files (training data).
    """
    param  = JSON.parsefile("../hyper-parameters-chengdu.json")
    links = Dict([])
    region = param["region"]
    harbin = SpatialRegion{Float64}("chengdu",
                                    region["minlon"], region["minlat"],
                                    region["maxlon"], region["maxlat"],
                                    region["cellsize"], region["cellsize"])
    fnames = readdir(jldpath)
    for fname in fnames
        println("saving $fname...")
        jldfile = joinpath(jldpath, fname)
        savetraindata(h5path, links, harbin, jldfile)
    end
end
