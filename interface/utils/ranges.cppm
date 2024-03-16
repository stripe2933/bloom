export module bloom:utils.ranges;

import std;

#define FWD(...) static_cast<decltype(__VA_ARGS__) &&>(__VA_ARGS__)
#define INDEX_SEQ(Is, N, ...) [&]<std::size_t... Is>(std::index_sequence<Is...>) __VA_ARGS__ (std::make_index_sequence<N>{})

namespace utils::ranges{
    // TODO.CXX23: use built-in when available.
    template <typename Self>
    struct range_adaptor_closure {
        template <std::ranges::range R>
        friend constexpr auto operator|(R &&r, const Self &self) noexcept(std::is_nothrow_invocable_v<Self, R>) {
            return self(FWD(r));
        }
    };

namespace views{
    template <std::unsigned_integral T>
#if __cpp_lib_ranges_enumerate >= 202302L
    [[deprecated("Use std::views::enumerate instead.")]]
#endif
    struct enumerate_fn : range_adaptor_closure<enumerate_fn<T>>{
        template <std::ranges::input_range R>
        constexpr auto operator()(R &&r) const {
            return std::views::zip(std::views::iota(T { 0 }), FWD(r));
        }
    };

    template <std::size_t N>
#if __cpp_lib_ranges_zip >= 202110L
    [[deprecated("Use std::views::adjacent instead.")]]
#endif
    struct adjacent_fn : range_adaptor_closure<adjacent_fn<N>>{
        template <std::ranges::forward_range R>
        constexpr auto operator()(R &&r) const {
            return INDEX_SEQ(Is, N, {
                return std::views::zip(r | std::views::drop(Is)...);
            });
        }
    };
}

    template <typename TupleLike>
    struct to_tuple_like_fn : range_adaptor_closure<to_tuple_like_fn<TupleLike>> {
        template <std::ranges::input_range R>
        constexpr auto operator()(R &&r) const {
            constexpr std::size_t tuple_size = std::tuple_size_v<TupleLike>;
            if constexpr (std::ranges::sized_range<R>) {
                if (r.size() != tuple_size) {
                    throw std::invalid_argument { "Range size must be equal to the tuple size." };
                }
            }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
            return INDEX_SEQ(Is, std::tuple_size_v<TupleLike>, {
                auto it = r.begin();
                return TupleLike { (Is, *(it++))... };
            });
#pragma clang diagnostic pop
        }
    };
}

export namespace utils{
namespace ranges{
namespace views{
    template <std::unsigned_integral T = std::size_t>
    constexpr enumerate_fn<T> enumerate;

    template <std::size_t N>
    constexpr adjacent_fn<N> adjacent;
    constexpr adjacent_fn<2> pairwise;
}

    template <typename T, std::size_t N>
    constexpr to_tuple_like_fn<std::array<T, N>> to_array;
}
    namespace views = ranges::views;
}