var subset = function(s, k, a, df, field, value) {
	return k(s, _.filter(df, function(d) { return d[field] === value }));
};