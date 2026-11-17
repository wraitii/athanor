(function() {
    /**
     * @param {Function} target
     * @param {*} arrayLike
     * @returns {Array<*>}
     */
    function argumentList(target, arrayLike) {
        if (typeof target !== "function")
            throw new Error("apply requires a callable receiver");
        /** @type {Array<*>} */
        const args = [];
        if (arrayLike == null)
            return args;
        if (typeof arrayLike !== "object" && typeof arrayLike !== "function")
            throw new Error("apply requires an array-like argument list");
        const length = +arrayLike.length;
        if (length > 1048576)
            throw new Error("apply argument list exceeds the engine limit");
        for (let i = 0; i < length - length % 1; ++i)
            args[i] = arrayLike[i];
        return args;
    }

    /**
     * @this {Function}
     * @param {*} thisArg
     * @param {*} arrayLike
     * @returns {*}
     */
    $defineHidden(Function.prototype, "apply", function(thisArg, arrayLike) {
        if ($isOrdinaryArray(arrayLike)) {
            if (typeof this !== "function")
                throw new Error("apply requires a callable receiver");
            if (arrayLike.length > 1048576)
                throw new Error("apply argument list exceeds the engine limit");
            return $callSpread(this, thisArg, arrayLike);
        }
        return $callSpread(this, thisArg, argumentList(this, arrayLike));
    });

    $defineHidden(Function.prototype, "call", function(thisArg) {
        if (typeof this !== "function")
            throw new Error("call requires a callable receiver");
        const args = [];
        for (let i = 1; i < arguments.length; ++i)
            args[i - 1] = arguments[i];
        return $callSpread(this, thisArg, args);
    });
})();
