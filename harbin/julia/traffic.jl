
include("SpatialRegion.jl")
include("Trip.jl")
include("util.jl")
include("Link.jl")
using Statistics

function createflowtensor!(region::SpatialRegion,
                           trips::Vector{Trip})

    """
    Create the tensor that counts the number of taxi inflowing and outflowing each
    cell in `region` using the `trips`.

    region.I counts the inflow
    region.O counts the outflow
    region.S stores the mean speed
    region.C counts the speed
    """
    function normalizex!(X)
        X ./= sum(X)
        X ./= maximum(X)
    end

    reset!(region)
    for trip in trips
        fine_trip, _, v̄ = linear_interpolate(gps2webmercator.(trip.lon, trip.lat), trip.tms)
        for i = 2:length(fine_trip)
            px, py = coord2regionOffset(region, fine_trip[i-1][1:2]...) .+ 1
            cx, cy = coord2regionOffset(region, fine_trip[i][1:2]...) .+ 1
            if cx ≠ px || cy ≠ py
                region.O[py, px] += 1 # outflow
                region.I[cy, cx] += 1 # inflow
                region.S[cy, cx] += v̄[i] # speed
                region.C[cy, cx] += 1
            elseif v̄[i] ≠ v̄[i-1] # mean speed changes so we count it
                region.S[cy, cx] += v̄[i] # speed
                region.C[cy, cx] += 1
            end
        end
    end
    normalizex!(region.I)
    normalizex!(region.O)
    idx = region.C .> 0
    region.S[idx] ./= region.C[idx]
end

# harbin = SpatialRegion{Float64}("harbin",
#                                 126.506130, 45.657920,
#                                 126.771862, 45.830905,
#                                 200., 200.)
# reset!(harbin)


function createflowtensorlink!(links::Dict,
                           trips::Vector{Trip}, trip_indices::Vector)
    """
    Create the tensor that counts the number of taxi inflowing and outflowing each
    road link using the `trips`.

    link.I counts the inflow
    link.O counts the outflow
    link.S stores the mean speed
    link.C counts the speed
    """
    function normalizex!(X)
        X ./= sum(X)
        X ./= maximum(X)
    end
    
    max_I = 0
    max_O = 0
    
    for idx = 1: length(trips) 
        trip = trips[idx]
        start_idx = trip_indices[idx][1]
        end_idx = trip_indices[idx][2]
        prev_link_id = trip.roads[trip.indices[start_idx] + 1]
        for i = start_idx + 1:end_idx
            if trip.spdist[i] > 0.1
                println(string(trip.roads))
                println(string(trip.lon))
                println(string(trip.lat))
                println(string(trip.tms))
                println(string(trip.spdist))
            end
            v̄ = trip.spdist[i] / ((trip.tms[i] - trip.tms[i - 1]) / 60.0)
            # remember that julia uses 1-based index
            for j = trip.indices[i - 1] + 1:trip.indices[i] + 1
                if haskey(links, trip.roads[j])
                    current_link = get(links, trip.roads[j], 0)
                else
                    current_link = Link(trip.roads[j])
                    links[trip.roads[j]] = current_link
                end
        
                current_link = get(links, trip.roads[j], 0)
                current_link.S += v̄ # speed
                current_link.C += 1

                if trip.roads[j] ≠ prev_link_id
                    prev_link = get(links, prev_link_id, 0)
                    current_link.I += 1 # inflow of current trip increase by 1
                    if current_link.I > max_I
                        max_I = current_link.I
                    end
                    prev_link.O += 1 # outflow of previous trip increase by 1
                    if prev_link.O > max_O
                        max_O = prev_link.O
                    end
                end
                prev_link_id = trip.roads[j]
            end
        end
        
    end
    
    max_S = 0
    min_S = 1
    for (gid, link) in links
        if link.C > 0
            link.S /= link.C
            if link.S > max_S
                max_S = link.S
            end
            if link.S < min_S
                min_S = link.S
            end
        end
        # normalize data
#         link.I /= 545.0
#         link.O /= 539.0
#         link.S /= 0.8167545199394226
    end
#     link.I /= max_I
#     link.O /= max_O
#     link.S /= max_S
#     println("current slot max_S & min_S: " * string(max_S) * " " * string(min_S))
    [max_I, max_O, max_S]
end